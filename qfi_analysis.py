"""
Code to produce the plots seen in the Quantile Frailty Index Paper
(Stubbings et al. 2021)

Will require Data in the correct format (unclear if provided in GitHub
Repository)

"""
import pylab as pl
import pandas as pd
import scipy.stats as stats
import operator
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from qfi_functions import *

if __name__ == "__main__":

    # close any existing plots
    pl.close('all')
    
    # Data set to look at
    data_set = "CSHA"
    # Wave to look at if doing elsa data
    wave = 4

    # Data directory
    data_dir = "../ELSA/Data/"
    plots_dir = "Plots/"

    mortality_followup = 5 # years
    condition_age = 35

    # Loading data
    # Data sets are not fully standardized so there are some gory details
    # will depend on your data as well
    # Basically we want:
    #           Array of biomarker measurements
    #           1D arrays of sex, age, mortalityality (binary at n years),
    #           if it's available also have fi-lab (thresholds from previous
    #           publications) and fi-clin (also not necessarily around)
    if "ELSA" in data_set:
        file_name = data_dir + "{0}Wave{1}Data.csv".format(data_set, wave)
        data_set += "Wave{}".format(wave)
        biomarker_columns = [5, -1]
        data_df = pd.read_csv(file_name, index_col=0)
        data_df.dropna(axis = 1, how = 'all', inplace = True)
        biomarker_list = data_df.columns[
                            biomarker_columns[0]:biomarker_columns[1]]
        biomarkers = data_df[biomarker_list].values
        ages = data_df['age'].values
        sex = data_df['sex'].values + 1 # sex is encoded 0 male, 1 female in
        # ELSA, change it to 1, 2 like CSHA/NHANES
        # death ages: -1 if no observed mortalityality
        death_ages = data_df['death age'].values
        # get death ages
        time_to_death = death_ages - ages
        # set no death events to be -1
        time_to_death[death_ages == -1] = -1
        # encode mortalityality as binary event at follow up
        mortality = (time_to_death < mortality_followup) & (
            time_to_death != -1)
        mortality = mortality.astype(int)

    elif "NHANES" in data_set:
        file_name = data_dir + "nhanes.csv"
        biomarker_columns = [0, -7]
        data_df = pd.read_csv(file_name, index_col=0)
        biomarker_list = data_df.columns[
                            biomarker_columns[0]:biomarker_columns[1]]
        biomarkers = data_df[biomarker_list].values
        ages = data_df['age'].values
        sex = data_df['sex'].values # sex is encoded 1 male, 2 female in NHANES
        time_to_death = data_df['TTD'].values # in months
        mortality = data_df['mort'].values # NHANES has binary 5 year mortality
        fi_pub = data_df['FILab'].values

    elif "CSHA" in data_set:
        file_name = data_dir + "csha_data.csv"
        biomarker_columns = [0, -6]
        data_df = pd.read_csv(file_name, index_col=0)
        biomarker_list = data_df.columns[
                            biomarker_columns[0]:biomarker_columns[1]]
        biomarkers = data_df[biomarker_list].values
        ages = data_df['Age'].values
        sex = data_df['Sex'].values # sex is encoded 1 male, 2 female in NHANES
        time_to_death = data_df['TimeTillDeath'].values # in days
        mortality_threshold = 365.25 * mortality_followup
        mortality = pl.full(len(time_to_death), 0)
        mortality[~pl.isnan(time_to_death)] = (
            time_to_death[~pl.isnan(time_to_death)] < mortality_threshold)
        mortality = mortality.astype(int)
        fi_pub = data_df['FILab'].values

    else:
        print("Pick a working dataset")
    

    conditions = spearman_age_conditions(biomarkers, ages, condition_age)

    # Demographics: Age quantilesribution
    plot_age_quantilesribution = 0
    if plot_age_quantilesribution:
        age_bins = pl.arange(20, 110, 5)
        pl.figure(figsize = (8,6))
        counts, bins = pl.hist(ages, age_bins)[:2]
        bins = bins[:-1]
        pl.xlabel('Age (years)', fontsize = fs)
        pl.ylabel('Count', fontsize = fs)
        filled_bins = bins[counts > 0]
        pl.xlim(min(filled_bins) - 1, max(filled_bins) + 6)
        pl.savefig(plots_dir + '{}AgeDistribution.pdf'.format(data_set))

    # Figure 2 in QFI paper: number of quantiles considered
    quantile_coarsening_plot = 0
    if quantile_coarsening_plot:
        reference_ages = [80, 85]
        low_age, high_age = reference_ages

        # ways of selecting biomarkers
        deficit_numbers = [5, 5, biomarkers.shape[1]]
        selection_types = ['Random', 'Best', 'All']

        # finding the best biomarkers
        biomarker_aucs = []
        for i in range(biomarkers.shape[1]):
            biomarker = biomarkers[:,i]
            auc = roc_auc_score(mortality[~pl.isnan(biomarker)],
                        biomarker[~pl.isnan(biomarker)])
            if auc < 0.5:
                auc = 1 - auc
            biomarker_aucs.append(auc)

        biomarker_aucs = pl.asarray(biomarker_aucs)
        rankings = pl.searchsorted(pl.sort(-biomarker_aucs), -biomarker_aucs)
        best_biomarkers = pl.where(rankings < 5)[0]
        #print(pl.asarray(biomarker_labels)[best_biomarkers])

        # setup stacked figure
        fig, ax = pl.subplots(3, 1, figsize = (8, 6), gridspec_kw={
            "height_ratios": [1,1.5,2]}, sharex = True)
        pl.subplots_adjust(hspace = 0, wspace = 0.05)
        
        # sampling: n_samp is number of resamples per biomarker set
        # num selections is how many times to re-chose the biomarkers
        n_samp = 20
        num_selections = 20
        selection_type = 'Random'

        # biggest group resolution (for QFI calculation)
        N_tot = pl.shape(biomarkers)[0]
        Ns = [2,3,4,5,6,7,8,9,10,N_tot]
        # ticks for plotting
        N_ticks = [n for n in Ns]
        N_ticks[-1] = 11

        # calculating full population values
        age_mask = (ages >= low_age) & (ages < high_age)
        reference_biomarkers = biomarkers[age_mask,:]
        quantiles = projected_quantiles(biomarkers, reference_biomarkers,
                                        conditions)
        qfi_full = pl.nanmean(quantiles, axis = 1)
        auc_full = roc_auc_score(mortality, qfi_full)

        # sampling time
        for n, num_deficits in enumerate(deficit_numbers):
            aucs = []
            # boxplot style data visualization
            means = []
            upper_99 = []
            upper_25 = []
            lower_25 = []
            lower_99 = []
            # going over the number of quantiles
            for N in Ns:
                # calculating full data numbers
                coarse_quantiles = resolution(quantiles, N)
                coarse_qfi = pl.nanmean(coarse_quantiles, axis = 1)
                auc = roc_auc_score(mortality, coarse_qfi)
                aucs.append(auc)
                # resampling with half the individuals
                resampled_aucs = []
                for i in range(num_selections):
                    # middle plot is best biomarkers
                    if n == 1:
                        deficit_ids = best_biomarkers
                    # top plot is random biomarkers
                    else:
                        deficit_ids = pl.choice(pl.arange(biomarkers.shape[1]),
                            size = num_deficits, replace = False)
                    # select the biomarkers 
                    biomarkers_selection = biomarkers[:, deficit_ids]
                    # select relevant risk directions
                    conditions_selection = pl.asarray(conditions)[deficit_ids]
                    # have to check for fully missing data again
                    missing_biomarkers = pl.sum(pl.isnan(biomarkers_selection),
                                                axis = 1)
                    # remove individuals missing too much data
                    mortality_selection = mortality[
                                            missing_biomarkers < num_deficits]
                    biomarkers_selection = biomarkers_selection[
                                        missing_biomarkers < num_deficits, :]
                    # get the age reference biomarkers
                    ages_selection = ages[missing_biomarkers < num_deficits]
                    age_mask = (ages_selection >= low_age) & (
                                ages_selection < high_age)
                    reference_biomarkers_selection = biomarkers_selection[
                                                            age_mask,:]
                    # resample the population and calculate AUCS
                    selection_aucs = qfi_reference_bootstrap(
                        biomarkers_selection, reference_biomarkers_selection,
                        mortality_selection, conditions_selection, n_samp = n_samp,
                        n_quantiles = N)
                    resampled_aucs += selection_aucs
                # record data on AUC quantilesribution
                means.append(pl.average(resampled_aucs))
                sorted_aucs = pl.sort(resampled_aucs)
                size = len(sorted_aucs)
                upper_99.append(sorted_aucs[-int(0.01*size) - 1])
                lower_99.append(sorted_aucs[int(0.01 * size)])
                upper_25.append(sorted_aucs[-int(0.25 * size) - 1])
                lower_25.append(sorted_aucs[int(0.25 * size) - 1])

            # plotting
            p = ax[n]#int(n>1), int(n%2)]
            p.fill_between(N_ticks, lower_25, upper_25, facecolor = 'C0',
                alpha = 0.3, label = 'Bootstrapped AUC Quartiles')
            if n > 2:
                p.plot(N_ticks, aucs, 'k-', label = 'Full Data')
            p.plot(N_ticks, means, 'k--', label = 'Bootstrapped Mean')
            p.plot(N_ticks, upper_99, 'k:')
            p.plot(N_ticks, lower_99, 'k:',
                label  = 'Bootstrapped 99th Percentiles')
            #pl.legend(loc = 'lower right', frameon = False)
            p.set_xticks(N_ticks)
            p.set_xticklabels(labels = Ns[:-1] + ['QFI'])
            #pl.xscale('log')
            #pl.xlabel('Number of Quantiles')
            #pl.ylabel('AUC')
            if n == 2:
                x_pos = 0.08
            else:
                x_pos = 0.6
            p.annotate('{0} {1} Biomarkers'.format(selection_types[n],
                                                num_deficits),
                xy = (x_pos, 0.02),
                va = 'bottom', ha = 'left', xycoords = 'axes fraction',
                fontsize = fs)
        
        pl.legend(loc = 'lower right')
        #ax[0,0].annotate(study_name + ' Data', xy = (0.01, 1.0), va = 'bottom',
        #    xycoords = 'axes fraction', fontsize = fs)
        pl.annotate('Number of Risk Categories', xy = (0.55, 0.03),
            ha = 'center', xycoords = 'figure fraction', fontsize = fs)
        pl.annotate('AUC', xy = (0.05, 0.5), rotation = 'vertical',
            va = 'center', ha = 'center', xycoords = 'figure fraction',
            fontsize = fs)
        pl.savefig(plots_dir + 
            '{0}QuantileCoarsening{1}.pdf'.format(data_set, date))

    # Figure 3: cross validation with GCP and age-reference resampling
    qfi_prediction_plot = 0
    if qfi_prediction_plot:
        fi_list = []
        fi_labels = []
        if "ELSA" not in data_set:
            fi_pub_auc = roc_auc_score(mortality, fi_pub)
        
        n_samp = 100
        cross_list = []
        full_set_aucs = []
        labels = []

        age_references = [[20, 25], [50, 55], [80,85]][::-1]
        if 'CSHA' in data_set:
            age_references = [[65,70], [80,85], [95,100]] 
        if "ELSA" in data_set:
            age_references = [[55,60], [80,85], [90,95]]
        for reference_age in age_references:
            labels.append('QFI-{}'.format(reference_age[0]))

            reference_data = reference_population(biomarkers, reference_age,
                reference_ages = ages)
            cross_val_aucs = qfi_reference_bootstrap(biomarkers, reference_data,
                mortality, conditions, n_samp = n_samp)
            cross_list.append(cross_val_aucs)
            projected_data = projected_quantiles(biomarkers, reference_data, conditions)
            projected_fi = pl.nanmean(projected_data, axis = 1)
            fi_list.append(projected_fi)
            full_projected_auc = roc_auc_score(mortality, projected_fi)
            full_set_aucs.append(full_projected_auc)

        ###### QFI AUCS
        labels.append('QFI-All')
        quantiles = projected_quantiles(biomarkers, biomarkers, conditions)
        full_set_fi = pl.nanmean(quantiles, axis  = 1)
        fi_list.append(full_set_fi)
        full_set_auc = roc_auc_score(mortality, full_set_fi)
        full_set_aucs.append(full_set_auc)
        cross_val_aucs = qfi_reference_bootstrap(biomarkers, biomarkers,
                            mortality, conditions, n_samp = n_samp)
        cross_list.append(cross_val_aucs)
     
        ##### FI-GCP Calculatins
        cross_val_cutpoints = [0.5, 0.8]
        for i, cut in enumerate(cross_val_cutpoints):
            labels.append('FI-GCP-{0:.1f}'.format(cut))
            cross_scores = fi_gcp_cross_validation(biomarkers, mortality, cut,
                conditions, 'fraction',
                metric = 'AUC', samples = n_samp, subset_count = 2,
                mortality = 'mortality')
            cross_list.append(cross_scores)
            cuts = cutpoint_calculator(biomarkers, conditions, cut)
            full_set_fi = cutpoint_fi(biomarkers, cuts, conditions)
            fi_list.append(full_set_fi)
            gcp_fi_score = roc_auc_score(mortality, full_set_fi)

            full_set_aucs.append(gcp_fi_score)

        ### Tertile AUCS:
        """
        labels.append('Tertiles')
        cross_val_aucs = fi_q_bootstrap(biomarkers, mortality,
                        conditions, n_samp = n_samp, n_quantiles = 3)
        cross_list.append(cross_val_aucs)
        coarse_quantiles = resolution(quantiles, 3)
        fi_tertiles = pl.nanmean(coarse_quantiles, axis = 1)
        full_set_auc = roc_auc_score(mortality, fi_tertiles)
        full_set_aucs.append(full_set_auc)
        """


        ##### LOGISTIC REGRESSION CALCULATIONS
        """
        labels.append('Logistic\nRegression')
        # need to fill missing values: just use mean fill
        biomarkers_mean_fill = pl.copy(biomarkers)
        for d in range(pl.shape(biomarkers)[1]):
            col = pl.copy(biomarkers[:,d])
            mean = pl.nanmean(col)
            col[pl.isnan(col)] = mean
            biomarkers_mean_fill[:,d] = col

        n_samp = 10
        aucs = []
        #biomarkers_mean_fill = biomarkers[~pl.isnan(biomarkers).any(axis = 1)]
        #mort = mortality[~pl.isnan(biomarkers).any(axis=1)]
        ones = pl.ones_like(ages)
        biomarkers_mean_fill = pl.column_stack([biomarkers_mean_fill, ones])
        print(pl.shape(biomarkers_mean_fill))
        for i in range(n_samp):
            biomarkers_train, biomarkers_test, mortality_train, mortality_test = train_test_split(
                biomarkers_mean_fill, mortality, test_size = 0.5)
            logreg = sm.Logit(mortality_train, biomarkers_train)
            result = logreg.fit()
            params = result.params
            pred = logreg.predict(params, biomarkers_test)
            aucs.append(roc_auc_score(mortality_test, pred))

        cross_list.append(aucs)
        
        const = pl.ones(len(mortality))

        logreg = sm.Logit(mortality,biomarkers_mean_fill)
        result = logreg.fit()
        params = result.params
        pred = logreg.predict(params, biomarkers_mean_fill)
        full_logreg_auc = roc_auc_score(mortality, pred)
        full_set_aucs.append(full_logreg_auc)
        """
        fig, ax = pl.subplots(figsize = (12,6))
        if "ELSA" not in data_set:
            ax.axhline(fi_pub_auc, ls = ':', color = 'C7',# label = 'Blodgett 2017',
                zorder = 1, lw = 5)
        ax.boxplot(cross_list, showmeans = True, labels = labels, whis = [1,99],
            meanline = True,
            meanprops = dict(linestyle = '-', color = 'k', alpha = 1,
                linewidth = 1.7, label = 'Mean'),
            medianprops = dict(linestyle = '--', color = 'k', alpha = 1,
                linewidth = 1.4, label = 'Median'))

        pl.ylabel('AUC', fontsize = fs)
        ax.tick_params(top = True, right = True, labelsize = fs)
        ax.grid(alpha = 0.3)
        ax.plot(pl.arange(len(full_set_aucs)) + 1, full_set_aucs, 'kD',
            markersize = 10, label = 'Full Dataset\nCalculation', zorder = 10)
        #ax.annotate(study_name, xy = (0.99, 0.01), xycoords = 'axes fraction',
        #    va = 'bottom', fontsize = fs, ha = 'right')
        if "NHANES" in data_set:
            ax.annotate('Blodgett 2017', xy = (4.45, fi_pub_auc),
                va = 'bottom', fontsize = fs*1.4, ha = 'right', color = 'C7')
        elif "CSHA" in data_set:
            ax.annotate('Howlett 2014', xy = (4.45, fi_pub_auc - 0.001),
                va = 'top', fontsize = fs*1.4, ha = 'right', color = 'C7')



        ###### Logistic regression on all the FIs included in the analysis.
        """
        fi_list.append(pl.ones(len(mortality)))
        fi_array = pl.asarray(fi_list).T
        logreg = sm.Logit(mortality, fi_array)
        result = logreg.fit()
        params = result.params
        pred = logreg.predict(params, fi_array)
        multiple_fi_auc = roc_auc_score(mortality, pred)
        ax.axhline(multiple_fi_auc, ls = '-.', color = 'k',
            label = 'Logreg of all FIs')
        """
        h, l = format_legend(ax)
        if "NHANES" in data_set:
            ax.legend(h[::-1], l[::-1], frameon = False, fontsize=fs,
                loc = 'upper right')#,
            #    bbox_to_anchor= (0.45, 0.03))


        fig.savefig(plots_dir + 
            '{0}CrossValidation{1}.pdf'.format(data_set, date))

    # Figure 4: QFI vs FI-Clin and Age (any data set)

    # Figure 4: ELSA-Specific Plotting Stuff: Diagnoses etc.

    # Figure 5: Limits of QFI with changing age reference
    # No Published FI-Lab to compare for ELSA data
    qfi_limits = 1
    if qfi_limits:
        bin_width = 5
        min_age = min(ages) - min(ages)%bin_width
        max_age = max(ages)
        age_bins = pl.arange(min_age, max_age + bin_width, bin_width)

        age_references = [[a, a+bin_width] for a in age_bins][:-1]
        age_points = age_bins[:-1] + bin_width/2
        aucs = []

        top_100 = []
        top_99 = []
        bot_0 = []
        bot_1 = []
        means = []
        reference_sizes = []
        colors = ['skyblue', 'royalblue'][::-1]
        num = int(0.01*pl.shape(biomarkers)[0])

        for reference_age in age_references:

            reference_data = reference_population(biomarkers, reference_age,
                reference_ages = ages)
            print(reference_age, reference_data.shape)
            reference_sizes.append(pl.shape(reference_data)[0])
            projected_data = projected_quantiles(biomarkers, reference_data,
                                                conditions)
            projected_fi = pl.nanmean(projected_data, axis = 1)
            sorted_fi  = pl.sort(projected_fi)
            top_100.append(sorted_fi[-1])
            top_99.append(sorted_fi[-num])
            bot_0.append(sorted_fi[0])
            bot_1.append(sorted_fi[num])
            means.append(pl.average(projected_fi))
            auc = roc_auc_score(mortality, projected_fi)
            aucs.append(auc)

        quantiles = projected_quantiles(biomarkers, biomarkers, conditions) 
        full_set_fi = pl.nanmean(quantiles, axis  = 1)
        full_set_auc = roc_auc_score(mortality, full_set_fi)
        pl.figure(figsize=(8,6))
        pl.axhline(full_set_auc, ls = ':', c = 'k', lw = 3,
            label = 'Full Population Reference')
        #pl.annotate(study_name + ' Data', xy = (0.01, 0.99), va = 'top',
        #    xycoords = 'axes fraction', fontsize = fs)
     
        pl.plot(age_points, aucs)
        pl.xlabel('Age of Reference Population (5 Year Bins)', fontsize = fs)
        pl.ylabel('AUC', fontsize = fs)
        # pl.savefig('plots/{0}AgeReferencePrediction.pdf'.format(data_set))

        pl.figure(figsize=(8,6))
        pl.fill_between(age_points, top_99, top_100, color = colors[0])
        pl.fill_between(age_points, bot_0, bot_1, color = colors[1])
        pl.plot(age_points, means, 'k:', label = 'mean')
        pl.xlabel('Age of Reference Population (5 Year Bins)', fontsize = fs)
        pl.ylabel('FI', fontsize = fs)
        pl.ylim(-0.02, 1.02)
        #pl.annotate(study_name + ' Data', xy = (0.01, 0.99), va = 'top',
        #    xycoords = 'axes fraction', fontsize = fs)

        # Published stuff (make it like 5 years wide or something)
        have_pub = 0
        if "NHANES" in data_set:
            name = "Blodgett"
            have_pub = 1
        elif "CSHA" in data_set:
            name = "Howlett"
            have_pub = 1
        if have_pub:
            pub_sorted = pl.sort(fi_pub)
            pub_ages = [age_points[-1] + 2.5, age_points[-1] + 7.5]

            pub_colors = ['firebrick', 'salmon']
            pl.fill_between(pub_ages, 2*[pub_sorted[-num]], 2*[pub_sorted[-1]],
                color =  pub_colors[1])
            pl.fill_between(pub_ages, 2*[pub_sorted[0]], 2*[pub_sorted[num]],
                color = pub_colors[0], edgecolor = pub_colors[0])
            pl.plot(pl.average(pub_ages), pl.average(fi_pub), marker = 'o',
                color = pub_colors[0])
            pl.xlim(min(age_bins), age_points[-1] + 7.5)
            pl.tick_params(right = True)
            pl.annotate(name, xy = (pl.average(pub_ages), pl.amax(fi_pub) + 0.05),
                ha = 'center', va = 'bottom', fontsize = fs * 1.2,
                rotation = 'vertical', color = pub_colors[1])
        pl.savefig(plots_dir + 
            '{0}AgeReferenceQFILimits.pdf'.format(data_set))

        pl.figure(figsize=(8,6))
        pl.plot(age_points, reference_sizes)
        pl.xlabel('Age of Reference Population (5 Year Bins)', fontsize = fs)
        pl.ylabel('Number of Individuals in Reference Population', fontsize = fs)
        #pl.annotate(study_name + ' Data', xy = (0.01, 0.99), va = 'top',
        #    xycoords = 'axes fraction', fontsize = fs)
        # pl.savefig('plots/{0}AgeReferenceNumbers.pdf'.format(data_set))


    # Figure 6: age-controlled prediction plot
    # here hard coded for QFI-80
    # There are a million different things to fit and measure here
    age_paired_prediction = 0
    if age_paired_prediction:
        bin_width = 5
        n_samp = 100
        weighted = True

        min_age = min(ages) - min(ages)%bin_width
        max_age = max(ages)
        age_bins = pl.arange(min_age, max_age + bin_width, bin_width)
        aucs = []
        labels = []


        age_mask = (ages >= 80) & (ages < 85)
        age_biomarkers = biomarkers[age_mask, :]
        age_quantiles = projected_quantiles(biomarkers, age_biomarkers, conditions)
        qfi_80 = pl.nanmean(age_quantiles, axis = 1)
        qfi_80_auc = roc_auc_score(mortality, qfi_80)
        aucs.append(qfi_80_auc)
        labels.append('QFI-80')


        # calculate the age-averaged auc from QFI-80
        N = len(mortality)
        avg_auc = age_averaged_prediction(qfi_80, mortality, ages,
                            bin_years = bin_width, weighted = weighted)
        aucs.append(avg_auc)
        labels.append('QFI-80 Age\nAveraged')

        logreg_list = [qfi_80, ages, pl.ones_like(ages)]
        logreg_array = pl.asarray(logreg_list).T
        logreg = sm.Logit(mortality, logreg_array)
        result = logreg.fit(disp=0)
        print(result.summary())
        params = result.params
        pred = logreg.predict(params, logreg_array)
        combined_qfi_80_auc = roc_auc_score(mortality, pred)
        aucs.append(combined_qfi_80_auc)
        labels.append('QFI-80 +\nAge LogReg')


        age_paired_fi = age_paired_qfi(biomarkers, ages, conditions, bin_width)
        age_paired_fi_auc = roc_auc_score(mortality, age_paired_fi)
        aucs.append(age_paired_fi_auc)
        labels.append('Age Paired\nQFI')

        logreg_list = [age_paired_fi, ages, pl.ones_like(ages)]
        logreg_array = pl.asarray(logreg_list).T
        logreg = sm.Logit(mortality, logreg_array)
        result = logreg.fit(disp=0)
        params = result.params
        #print(result.summary())
        pred = logreg.predict(params, logreg_array)
        combined_fi_age_paired_auc = roc_auc_score(mortality, pred)
        aucs.append(combined_fi_age_paired_auc)
        labels.append('Age-Paired +\nAge LogReg')

        biomarkers_mean_fill = pl.copy(biomarkers)
        for d in range(pl.shape(biomarkers_mean_fill)[1]):
            col = pl.copy(biomarkers[:,d])
            mean = pl.nanmean(col)
            col[pl.isnan(col)] = mean
            biomarkers_mean_fill[:,d] = col

        ones = pl.ones_like(ages)
        biomarkers_mean_fill = pl.column_stack([biomarkers_mean_fill, ones])

        logreg = sm.Logit(mortality, biomarkers_mean_fill)
        result = logreg.fit(disp=0)
        params = result.params
        pred = logreg.predict(params, biomarkers_mean_fill)
        aucs.append(roc_auc_score(mortality, pred))
        labels.append('Biomarker\nLogReg')

        avg_auc = age_averaged_prediction(pred, mortality, ages,
                            bin_years = bin_width, weighted = weighted)

        aucs.append(avg_auc)
        labels.append("Biomarker\nLogreg\nAge Averaged")

        biomarkers_mean_fill = pl.column_stack([biomarkers_mean_fill, ages])
        logreg = sm.Logit(mortality, biomarkers_mean_fill)
        result = logreg.fit(disp=0)
        params = result.params
        pred = logreg.predict(params, biomarkers_mean_fill)
        aucs.append(roc_auc_score(mortality, pred))
        labels.append('Biomarkers + \nage LogReg')


        #### CROSS balidation time.
        # do it all at once 
        cross_list = [[] for i in range(8)]
        for n in range(n_samp):
               
            split_data = train_test_split(biomarkers, mortality, ages,
                                                test_size = 0.5)
            train_biomarkers, test_biomarkers = split_data[:2]
            mortality_train, mortality_test = split_data[2:4]
            age_train, age_test = split_data[4:]

            age_mask = (age_test >= 80) & (age_test < 85)
            age_biomarkers = test_biomarkers[age_mask, :]
            age_quantiles = projected_quantiles(test_biomarkers,
                        age_biomarkers, conditions)

            qfi = pl.nanmean(age_quantiles, axis = 1)
            cross_list[0].append(roc_auc_score(mortality_test, qfi))
     
            # calculate the age-averaged auc from QFI-80
            N = len(mortality_test)
            avg_auc = age_averaged_prediction(qfi, mortality_test, age_test,
                            bin_years = bin_width, weighted = weighted)

            cross_list[1].append(avg_auc)
            #labels.append('QFI-80 Age\nAveraged')

            logreg_list = [qfi, age_test, pl.ones_like(age_test)]
            logreg_array = pl.asarray(logreg_list).T
            logreg = sm.Logit(mortality_test, logreg_array)
            result = logreg.fit(disp=0)
            #print(result.summary())
            params = result.params
            pred = logreg.predict(params, logreg_array)
            combined_qfi_80_auc = roc_auc_score(mortality_test, pred)
            cross_list[2].append(combined_qfi_80_auc)
            #labels.append('QFI-80 +\nAge LogReg')


            age_paired_fi = age_paired_qfi(test_biomarkers, age_test, conditions, bin_width)
            age_paired_fi_auc = roc_auc_score(mortality_test, age_paired_fi)
            cross_list[3].append(age_paired_fi_auc)
            #labels.append('Age Paired')

            logreg_list = [age_paired_fi, age_test, pl.ones_like(age_test)]
            logreg_array = pl.asarray(logreg_list).T
            logreg = sm.Logit(mortality_test, logreg_array)
            result = logreg.fit(disp=0)
            params = result.params
            #print(result.summary())
            pred = logreg.predict(params, logreg_array)
            combined_fi_age_paired_auc = roc_auc_score(mortality_test, pred)
            cross_list[4].append(combined_fi_age_paired_auc)
            #labels.append('Age-Paired +\nAge LogReg')

            ## Raw biomarkers
            biomarkers_mean_fill = pl.copy(test_biomarkers)
            for d in range(pl.shape(biomarkers_mean_fill)[1]):
                col = pl.copy(test_biomarkers[:,d])
                mean = pl.nanmean(col)
                col[pl.isnan(col)] = mean
                biomarkers_mean_fill[:,d] = col

            ones = pl.ones_like(age_test)
            biomarkers_mean_fill = pl.column_stack([biomarkers_mean_fill, ones])

            logreg = sm.Logit(mortality_test, biomarkers_mean_fill)
            result = logreg.fit(disp=0)
            params = result.params
            pred = logreg.predict(params, biomarkers_mean_fill)
            cross_list[5].append(roc_auc_score(mortality_test, pred))

            avg_auc = 0.0
            avg_auc = age_averaged_prediction(pred, mortality_test, age_test,
                            bin_years = bin_width, weighted = weighted)

            cross_list[6].append(avg_auc)

            ones = pl.ones_like(age_test)
            biomarkers_mean_fill = pl.column_stack([biomarkers_mean_fill, age_test])

            logreg = sm.Logit(mortality_test, biomarkers_mean_fill)
            result = logreg.fit(disp=0)
            params = result.params
            pred = logreg.predict(params, biomarkers_mean_fill)
            cross_list[7].append(roc_auc_score(mortality_test, pred))

        # Calculation order: QFI-80: 0, QFI-80 AgeAvg: 1, QFI+AgeLogreg: 2,
        # Age-Paired: 3, Age-Paired+AgeLogreg: 4, Biomarkers:5, Biomarkers AgeAvg:
        # 6, Biomarkers+Age:7

        # Desired order: QFI80 0, Biomarkers 1, QFI Age-Paired 2, QFI AgeAvg 3,
        # Biomarker AgeAvg 4, AgePaired+Age 5, QFI + Age 6, Biomarkers + age 7

        list_reorder = [0, 5, 3, 1, 6, 4, 2, 7]
        
        cross_list = list(pl.array(cross_list)[list_reorder])
        labels = list(pl.array(labels)[list_reorder])
        aucs = list(pl.array(aucs)[list_reorder])
        fig, ax = pl.subplots(1, 1, figsize = (15,7))
        ax.boxplot(cross_list, showmeans = True, labels = labels, whis = [1,99],
            positions = pl.arange(len(labels)),
            meanline = True,
            meanprops = dict(linestyle = '-', color = 'k', alpha = 1,
                linewidth = 1.7, label = 'Mean'),
            medianprops = dict(linestyle = '--', color = 'k', alpha = 1,
                linewidth = 1.4, label = 'Median'))

     
        ax.plot(pl.arange(len(labels)), aucs, 'kD', ms = 8,
            label = 'Full Dataset\nCalculation')
        ax.set_xticks(pl.arange(len(labels)))
        ax.set_xticklabels(labels = labels, fontsize = fs*0.9)
        ax.grid(alpha = 0.3)
        h, l = format_legend(ax)
        ax.legend(h[::-1], l[::-1], frameon = False, fontsize=fs)

        ax.set_ylabel('AUC', fontsize = fs)
        pl.savefig(plots_dir + 
            "{0}AgeControlledPrediction{1}.pdf".format(data_set,
                                        (1 - weighted) * 'Non' +'Weighted'))


    # Figure 7: sex-controlled QFI vs Age and QFI
    plot_stratified_qfi = 0
    if plot_stratified_qfi:
        sex_labels = ['Male', 'Female']
        sex_masks = [(sex == 1), (sex == 2)]
        vorders = [10, 1]

        values = [0, 0.2, 0.85, 1]
        cm = mpl.cm.viridis(pl.linspace(0, 1, 100))
        cm = mpl.colors.ListedColormap(cm)
        c_norm = pl.Normalize(vmin=0, vmax=values[-1])
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap = cm)
        colors = [scalar_map.to_rgba(v) for v in values]

        sex_colours = ['C0', colors[2]]
        sex_markers = ['o', 's']

        age_mask = (ages >= 80) & (ages < 85)
        quantiles_80 = projected_quantiles(biomarkers, biomarkers[age_mask,:], conditions)
        qqfi_80 = pl.nanmean(quantiles_80, axis = 1)
        bin_width = 5
        age_bins = pl.arange(min(ages), max(ages) + bin_width, bin_width)
        pl.figure(figsize = (8,6))
        for i, label in enumerate(sex_labels):
            sex_mask = sex_masks[i]
            both_mask = sex_mask & age_mask
            sex_biomarkers = biomarkers[sex_mask,:]
            reference_biomarkers = biomarkers[both_mask, :]




            adjusted_quantiles = projected_quantiles(sex_biomarkers,
                reference_biomarkers, conditions)
            adjusted_qfi = pl.nanmean(adjusted_quantiles, axis = 1)
            mid_points, binned_adjusted_qfi = bin_x_by_y(
                adjusted_qfi, ages[sex_mask], age_bins)
            non_adjusted_qfi = qqfi_80[sex_mask]
            mid_points, binned_qfi = bin_x_by_y(
            non_adjusted_qfi, ages[sex_mask], age_bins)

            size_threshold = [len(f) > min_count for f in binned_adjusted_qfi]
            adjusted_means = [pl.average(a) for i, a in
                enumerate(binned_adjusted_qfi) if size_threshold[i]]
            adjusted_errors = [pl.std(a)/pl.sqrt(len(a)) for i, a in
                enumerate(binned_adjusted_qfi) if size_threshold[i]]
            mid_points = list_logic(mid_points, size_threshold)
            pl.errorbar(mid_points, adjusted_means, yerr = adjusted_errors,
                c = sex_colours[i], marker = sex_markers[i], ls = 'none',
                capsize = 3, label = label + " Adjusted", ms = 7 + 2*i,
                zorder = vorders[i])

            non_adjusted_qfi = qqfi_80[sex_mask]
            mid_points, binned_qfi = bin_x_by_y(
                non_adjusted_qfi, ages[sex_mask], age_bins)

            size_threshold = [len(f) > min_count for f in binned_qfi]
            non_adjusted_means = [pl.average(a) for a in binned_qfi
                if len(a) > min_count]
            non_adjusted_errors = [pl.std(a)/pl.sqrt(len(a)) for a in
                binned_qfi if len(a) >  min_count]
            mid_points = list_logic(mid_points, size_threshold)
            pl.errorbar(mid_points, non_adjusted_means,
                yerr = non_adjusted_errors, c = sex_colours[i],
                marker = sex_markers[i], mfc = 'w', ls = 'none', capsize = 3,
                label = label + " Non-Adjusted", ms = 7 + 2*i,
                zorder = vorders[i])

            mid_points, binned_fi_clin = bin_x_by_y(
                fi_clin[sex_mask], ages[sex_mask], age_bins)
            fi_clin_means = [pl.average(a) for a in binned_fi_clin if 
                len(a) > min_count]
            fi_clin_errors = [pl.std(a)/pl.sqrt(len(a)) for a in
                binned_fi_clin if len(a) > min_count]
            mid_points = list_logic(mid_points, size_threshold)
            pl.errorbar(mid_points, fi_clin_means, yerr = fi_clin_errors,
                c = 'k', marker = sex_markers[i], ls = 'none',
                capsize = 3, label = label + " FI-Clin", ms = 8)


            """
            adjusted_means = [pl.average(a) for a in binned_adjusted_qfi]
            adjusted_errors = [pl.std(a)/pl.sqrt(len(a)) for a in
                binned_adjusted_qfi]
            pl.errorbar(mid_points, adjusted_means, yerr = adjusted_errors,
                c = sex_colours[i], marker = sex_markers[i], ls = 'none',
                capsize = 3, label = label + " Adjusted", ms = 7 + 2*i,
                zorder = vorders[i])

            non_adjusted_means = [pl.average(a) for a in binned_qfi]
            non_adjusted_errors = [pl.std(a)/pl.sqrt(len(a)) for a in
                binned_qfi]
            pl.errorbar(mid_points, non_adjusted_means,
                yerr = non_adjusted_errors, c = sex_colours[i],
                marker = sex_markers[i], mfc = 'w', ls = 'none', capsize = 3,
                label = label + " Non-Adjusted", ms = 7 + 2*i,
                zorder = vorders[i])

            mid_points, binned_fi_clin = bin_x_by_y(
                fi_clin[sex_mask], ages[sex_mask], age_bins)
            fi_clin_means = [pl.average(a) for a in binned_fi_clin]
            fi_clin_errors = [pl.std(a)/pl.sqrt(len(a)) for a in
                binned_fi_clin]
            pl.errorbar(mid_points, fi_clin_means, yerr = fi_clin_errors,
                c = 'k', marker = sex_markers[i], ls = 'none',
                capsize = 3, label = label + " FI-Clin", ms = 8)
            """

        pl.legend(fontsize = fs*0.7)
        pl.xticks(fontsize = fs*0.8)
        pl.yticks(fontsize = fs*0.8)
        pl.xlabel('Age (years)', fontsize = fs)
        pl.ylabel('QFI', fontsize = fs)
        pl.savefig('Plots/PaperPlots/{}SexAdjustedQFIVsAge.pdf'.format(data_set))

        ##### WORLDS SHITTIEST COPY PASTE

        bin_width = 0.1
        qfi_bins = pl.arange(0, 1 + bin_width, bin_width)
        pl.figure(figsize = (8,6))
        for i, label in enumerate(sex_labels):
            sex_mask = sex_masks[i]
            both_mask = sex_mask & age_mask
            sex_biomarkers = biomarkers[sex_mask,:]
            reference_biomarkers = biomarkers[both_mask, :]
            adjusted_quantiles = projected_quantiles(sex_biomarkers,
                reference_biomarkers, conditions)
            adjusted_qfi = pl.nanmean(adjusted_quantiles, axis = 1)

            mid_points, binned_adjusted_qfi = bin_x_by_y(fi_clin[sex_mask], 
                adjusted_qfi, qfi_bins)

            length_mask = [len(a) > 10 for a in binned_adjusted_qfi]
            adjusted_means = [pl.average(a) for a in binned_adjusted_qfi
                if len(a) > 10]
            adjusted_errors = [pl.std(a)/pl.sqrt(len(a)) for a in
                binned_adjusted_qfi if len(a) > 10]

            pl.errorbar(mid_points[length_mask], adjusted_means,
                yerr = adjusted_errors,
                c = sex_colours[i], marker = sex_markers[i], ls = 'none',
                capsize = 3, label = label + " Adjusted", ms = 7 + 2*i,
                zorder = vorders[i])


            non_adjusted_qfi = qqfi_80[sex_mask]
            mid_points, binned_qfi = bin_x_by_y(fi_clin[sex_mask],
                non_adjusted_qfi, qfi_bins)
            length_mask = [len(a) > 10 for a in binned_qfi]

            non_adjusted_means = [pl.average(a) for a in
                binned_qfi if len(a) > 10]
            non_adjusted_errors = [pl.std(a)/pl.sqrt(len(a)) for a in
                binned_qfi if len(a) > 10]
            pl.errorbar(mid_points[length_mask], non_adjusted_means,
                yerr = non_adjusted_errors, c = sex_colours[i],
                marker = sex_markers[i], mfc = 'w', ls = 'none', capsize = 3,
                label = label + " Non-Adjusted", ms = 7 + 2*i,
                zorder = vorders[i])

        pl.legend(fontsize = fs)
        pl.xticks(fontsize = fs*0.8)
        pl.yticks(fontsize = fs*0.8)
        pl.xlabel('QFI', fontsize = fs)
        pl.ylabel('FI-Clin', fontsize = fs)
        pl.savefig('Plots/PaperPlots/{0}SexAdjustedFIClinVsQFI.pdf'.format(
            data_set))

    # Figure 7: Cross-validation using sex-specific reference populations
    sex_specific_prediction = 0
    if sex_specific_prediction:
        age_bins = pl.arange(min(ages), max(ages) + 5, 5)
        
        reference_ages = [80, 85]
        sex_labels = ['Male', 'Female']
        sex_masks = [(sex == 1), (sex == 2)]
        age_mask = (ages >= reference_ages[0]) & (ages < reference_ages[1])

        aucs = []
        # Non adjusted (regular qfi)
        quantiles = projected_quantiles(biomarkers, biomarkers[age_mask,:], conditions)
        qfi = pl.nanmean(quantiles, axis = 1)
        aucs.append(roc_auc_score(mortality, qfi))
        # adjust for sex
        mortality_list = []
        sex_adjusted_qfi_list  = []
        fully_adjusted_qfi_list = []
        for sex_mask in sex_masks:
            # non-adjusted
            aucs.append(roc_auc_score(mortality[sex_mask], qfi[sex_mask]))
            # adjusted
            mortality_list.append(mortality[sex_mask])
            sex_biomarkers = biomarkers[sex_mask,:]
            adjusted_reference = biomarkers[age_mask & sex_mask,:]
            adjusted_quantiles = projected_quantiles(sex_biomarkers,
                adjusted_reference, conditions)
            adjusted_qfi = pl.nanmean(adjusted_quantiles, axis = 1)
            sex_adjusted_qfi_list.append(adjusted_qfi)
            aucs.append(roc_auc_score(mortality[sex_mask], adjusted_qfi))
            
            # Age and sex paired
            age_sex_qfi = age_paired_qfi(sex_biomarkers, ages[sex_mask], conditions,
                bin_width = 5)
            fully_adjusted_qfi_list.append(age_sex_qfi)
            aucs.append(roc_auc_score(mortality[sex_mask], age_sex_qfi))

        aucs.insert(1, roc_auc_score(pl.concatenate(mortality_list),
                                pl.concatenate(sex_adjusted_qfi_list)))
        aucs.insert(2, roc_auc_score(pl.concatenate(mortality_list),
                                pl.concatenate(fully_adjusted_qfi_list)))
        values = [0, 0.2, 0.85, 1]
        cm = mpl.cm.viridis(pl.linspace(0, 1, 100))
        cm = mpl.colors.ListedColormap(cm)
        c_norm = pl.Normalize(vmin=0, vmax=values[-1])
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap = cm)
        colors = [scalar_map.to_rgba(v) for v in values]

        sex_colours = ['C0', colors[2]]
        sex_markers = ['o', 's']
        n_samp = 100

        # want: Adjusted and non=adjusted AUCs for m/f, and total (6 elements)
        labels = ['QFI-80', 'QFI-80\nsex-paired', 'QFI-80 age+\nsex-paired',
            'Male QFI-80', 'Male QFI-80\nsex-paired',
            'Male QFI-80 age+\nsex-paired', 'Female QFI-80',
            'Female QFI-80\nsex-paired', 'Female QFI-80 age+\nsex-paired']
        labels = 3 * ['QFI-80', 'QFI-80\nSex-Paired', 'Age and Sex-\nPaired QFI']
        resampled_aucs = [[] for i in range(len(labels))]

        # resampling
        for n in range(n_samp):
            # sample half the data
            split_data = train_test_split(biomarkers, mortality, ages, sex,
                test_size = 0.5)
            # just need one half of the data
            sampled_biomarkers  = split_data[0]
            sampled_mort= split_data[2]
            sampled_age = split_data[4]
            sampled_sex = split_data[6]

            sex_masks = [(sampled_sex == 1), (sampled_sex == 2)]
            age_mask = (sampled_age >= reference_ages[0]) & (
                            sampled_age < reference_ages[1])

            # Non sex specific stuff (QFI 80)
            reference_biomarkers = sampled_biomarkers[age_mask, :]
            quantiles = projected_quantiles(sampled_biomarkers,
                reference_biomarkers, conditions)
            qfi = pl.nanmean(quantiles, axis = 1)
            resampled_aucs[0].append(roc_auc_score(sampled_mortality, qfi))

            # sex adjusted stuff
            sex_adjusted_qfis = []
            mortalities = []
            fully_adjusted_qfis = []
            for i, label in enumerate(sex_labels):
                sex_mask = sex_masks[i]
                mortality = sampled_mortality[sex_mask]
                mortalities.append(mortality)
                # Sex + 80-85 year old reference for QFI-80
                both_mask = sex_mask & age_mask
                sex_biomarkers = sampled_biomarkers[sex_mask,:]
                sex_reference_biomarkers = sampled_biomarkers[both_mask, :]
                sex_adjusted_quantiles = projected_quantiles(sex_biomarkers,
                    sex_reference_biomarkers, conditions)
                sex_adjusted_qfi = pl.nanmean(sex_adjusted_quantiles, axis = 1)
                sex_adjusted_qfis.append(sex_adjusted_qfi)
                resampled_aucs[4 + 3*i].append(roc_auc_score(mortality,
                    sex_adjusted_qfi))
                # Non-Adjusted qfi (just grabbed from full qfi quantilesribution)    
                non_adjusted_qfi = qfi[sex_mask]
                resampled_aucs[3 + 3*i].append(roc_auc_score(mortality,
                    non_adjusted_qfi))
                # Age and sex paired QFIs:
                age_sex_qfi = age_paired_qfi(sex_biomarkers,
                    sampled_age[sex_mask], conditions, bin_width = 5)
                fully_adjusted_qfis.append(age_sex_qfi)
                resampled_aucs[5+3*i].append(roc_auc_score(mortality,
                    age_sex_qfi))


            # Record auc for combined male and female adjusted QFIs
            resampled_aucs[1].append(roc_auc_score(
                pl.concatenate(mortalities),
                pl.concatenate(sex_adjusted_qfis)))
            resampled_aucs[2].append(roc_auc_score(
                pl.concatenate(mortalities),
                pl.concatenate(fully_adjusted_qfis)))
         
        # Plot on box and whisker like other crossvalidation/resampling stuff
        fig, ax = pl.subplots(1, 1, figsize = (8,6))
        ax.boxplot(resampled_aucs, showmeans = True, labels = labels,
            whis = [1,99],
            positions = pl.arange(len(labels)),
            meanline = True,
            meanprops = dict(linestyle = '-', color = 'k', alpha = 1,
                linewidth = 1.7, label = 'Mean'),
            medianprops = dict(linestyle = '--', color = 'k', alpha = 1,
                linewidth = 1.4, label = 'Median'))

        label_positions = pl.arange(len(labels))[1::3]
        category_labels = ['Combined M+F', 'Male Only', 'Female Only']
        max_auc = pl.amax(resampled_aucs)
        min_auc = pl.amin(resampled_aucs)
        ax.set_ylim(min_auc - .05, max_auc + 0.05)
        for i, x_pos in enumerate(label_positions):
            ax.annotate(category_labels[i], xy = (x_pos, max_auc + 0.01),
                ha = 'center', va = 'bottom', fontsize = fs)
        ax.plot(pl.arange(len(labels)), aucs, 'kD', ms = 6,
            label = 'Full Dataset\nCalculation')
        ax.set_xticks(pl.arange(len(labels)))
        ax.set_xticklabels(labels = labels, fontsize = fs*0.6,
            rotation = 25)
        ax.set_ylabel('AUC', fontsize = fs)
        h, l = format_legend(ax)
        ax.legend(h, l, frameon = False, loc = 'lower left')
        pl.savefig('Plots/PaperPlots/{}SexPrediction.pdf'.format(data_set))

    pl.show()
