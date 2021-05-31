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
    data_set = "ELSA"
    # Wave to look at if doing elsa data
    wave = 4

    # Data directory
    data_dir = "Data/"
    plots_dir = "Plots/"

    mortality_followup = 5 # years
    condition_age = 35
    min_count = 20

    # Loading data
    # Data sets are not fully standardized so there are some gory details
    # will depend on your data as well
    # Basically we want:
    #           Array of biomarker measurements
    #           1D arrays of sex, age, mortality (binary at n years),
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
        biomarker_labels = pl.asarray(biomarker_list)
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
        # Clinical frailty here is ADL/IADL deficits
        fi_clin = data_df['ADL/IADL'].values

        # Dataframe that contains number of new diagnoses at each wave
        num_new_diagnoses_df = pd.read_csv(
            data_dir + 'NumberOfNewDiagnoses.csv', index_col=0)
        # dataframe that shows which wave (if any) a specific diagnosis occured
        raw_diagnoses_df = pd.read_csv(
            data_dir + 'WaveOfFirstDiagnosis.csv', index_col=0)

        # Get the diagnosis data that overlaps with the biomarker data for this
        # wave (not guaranteed to match perfectly)
        matched_diagnosis_ids = num_new_diagnoses_df.id.isin(
                                                        data_df['id'].values)
        # reduce diagnoses dataframe
        matched_num_diagnoses_df = num_new_diagnoses_df.loc[
                                                matched_diagnosis_ids, :]
        # Match the biomarkers back the otherway (could be biomarkers with no
        # diagnoses data)
        matched_biomarker_ids = data_df.id.isin(
                                        matched_num_diagnoses_df['id'].values)
        matched_data_df = data_df.loc[matched_biomarker_ids,:]
        matched_biomarkers = matched_data_df[biomarker_list].values

        matched_diagnoses_waves = raw_diagnoses_df.loc[matched_diagnosis_ids,:]

    elif "NHANES" in data_set:
        file_name = data_dir + "nhanes.csv"
        biomarker_columns = [0, -7]
        data_df = pd.read_csv(file_name, index_col=0)
        biomarker_list = data_df.columns[
                            biomarker_columns[0]:biomarker_columns[1]]
        biomarkers = data_df[biomarker_list].values
        biomarker_labels = pl.asarray(biomarker_list)
        ages = data_df['age'].values
        sex = data_df['sex'].values # sex is encoded 1 male, 2 female in NHANES
        time_to_death = data_df['TTD'].values # in months
        mortality = data_df['mort'].values # NHANES has binary 5 year mortality
        fi_pub = data_df['FILab'].values
        fi_clin = data_df['FIClin'].values

    elif "CSHA" in data_set:
        file_name = data_dir + "csha_data.csv"
        biomarker_columns = [0, -6]
        data_df = pd.read_csv(file_name, index_col=0)
        biomarker_list = data_df.columns[
                            biomarker_columns[0]:biomarker_columns[1]]
        biomarkers = data_df[biomarker_list].values
        biomarker_labels = pl.asarray(biomarker_list)
        ages = data_df['Age'].values
        sex = data_df['Sex'].values # sex is encoded 1 male, 2 female in NHANES
        time_to_death = data_df['TimeTillDeath'].values # in days
        mortality_threshold = 365.25 * mortality_followup
        mortality = pl.full(len(time_to_death), 0)
        mortality[~pl.isnan(time_to_death)] = (
            time_to_death[~pl.isnan(time_to_death)] < mortality_threshold)
        mortality = mortality.astype(int)
        fi_pub = data_df['FILab'].values
        fi_clin = data_df['FICSHA'].values

    else:
        print("Pick a working dataset")
    

    conditions = spearman_age_conditions(biomarkers, ages, condition_age)

    # Figure 1: Rank normalization and reference cohort examples (in ELSA)
    gait_speed_example = 0
    if gait_speed_example:
        old = ((ages < 90) & (ages >= 80))
        young = ((ages < 70) & (ages >= 60))
        
        gait = biomarkers[:,2]

        gait_young = (gait[young])
        gait_young = gait_young[~pl.isnan(gait_young)]
        gait_young_x = pl.searchsorted(pl.sort(-gait_young), -gait_young)/(
            len(gait_young))
        gait_old = (gait[old])
        gait_old = gait_old[~pl.isnan(gait_old)]
        gait_old_x = pl.searchsorted(pl.sort(-gait_old), -gait_old)/(
            len(gait_old))

        values = [0, 0.2, 0.85, 1]
        colors = get_colours_from_cmap(values)
        
        c_old = 'C0'
        c_young = colors[2]
        
        pl.figure(figsize = (8,6))
        pl.plot(gait_young, gait_young_x, ls = '', color = c_young,
            marker = 'o')
        pl.plot(gait_old, gait_old_x, ls = '', color = c_old,
            marker = 'o')
        pl.ylabel('x', fontsize = fs)
        pl.xlabel('Gait Speed m/s', fontsize = fs)
        pl.savefig(plots_dir + 'gaitSpeedXExample.pdf')

        bw = 0.5/5
        bins = pl.arange(pl.nanmin(gait), pl.nanmax(gait) + bw, bw)
     

        pl.figure(figsize = (8,6))
        pl.hist(gait_young, bins, color = c_young, label = 'Ages [60, 70)')
        pl.hist(gait_old, bins, color = c_old, label = 'Ages [80, 90)')
        counts = pl.hist(gait_young, bins, edgecolor = c_young,
            facecolor = 'none',)[0]
        #    hatch = 'none')
        pl.legend(loc = 'upper left', fontsize = fs, frameon = False)
        pl.xlabel('Gait Speed m/s', fontsize = fs)
        pl.ylabel('Count', fontsize = fs)
        pl.ylim(0, 1.3*max(counts))
        pl.savefig(plots_dir + 'gaitSpeedHistogram.pdf')

        old_ages = ages[old]
        young_ages = ages[young]

        pl.figure(figsize = (8,6))
        age_bins = pl.arange(min(ages), max(ages) + 2, 2)
        pl.hist(young_ages, age_bins, color = c_young, label = 'Ages [60, 70)')
        pl.hist(old_ages, age_bins, color = c_old, label = 'Ages [80, 90)')
        pl.hist(ages, age_bins, facecolor = 'none', edgecolor = 'k')
        pl.xlabel('Age (years)', fontsize = 2*fs)
        pl.ylabel('Count', fontsize = 2*fs)
        pl.subplots_adjust(bottom = 0.15, left = 0.15)
        pl.xticks(fontsize = fs)
        pl.yticks(fontsize = fs)
        pl.savefig(plots_dir + 'ExampleAgeDistributionELSAWave2.pdf')


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
        label_pads = [10, 5, 0]

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

            auc_height = pl.amax(upper_99) - pl.amin(lower_99)
            font_height = 1/(2*(n + 2)) * auc_height * 1.05
            overlapping_aucs = lower_99[1:7]
            y_lim = [pl.amin([pl.amin(lower_99)*0.99, pl.amin(overlapping_aucs)
                                                    - font_height]),
                    pl.amax(upper_99)*1.01]
            p.set_ylim(y_lim)
            x_pos = 3
            y_pos = y_lim[0]
            p.annotate('{0} {1} Biomarkers'.format(selection_types[n],
                                                num_deficits),
                xy = (x_pos, y_pos),
                va = 'bottom', ha = 'left', 
                fontsize = fs)
        
        pl.legend(loc = 'lower right', frameon=True)
        #ax[0,0].annotate(study_name + ' Data', xy = (0.01, 1.0), va = 'bottom',
        #    xycoords = 'axes fraction', fontsize = fs)
        pl.annotate('Number of Risk Categories', xy = (0.55, 0.03),
            ha = 'center', xycoords = 'figure fraction', fontsize = fs)
        pl.annotate('AUC', xy = (0.05, 0.5), rotation = 'vertical',
            va = 'center', ha = 'center', xycoords = 'figure fraction',
            fontsize = fs)
        pl.savefig(plots_dir + 
            '{0}QuantileCoarsening.pdf'.format(data_set))

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
    plot_fi_clin_vs_qfi = 0
    if plot_fi_clin_vs_qfi:
        bin_width = 0.05
        fi_bins = pl.arange(0, 1+bin_width, bin_width)
        reference_ages = [80, 85]
        age_mask = (ages >= reference_ages[0]) & (
                        ages < reference_ages[1])
        age_reference_biomarkers = biomarkers[age_mask,:]
        age_reference_quantiles = projected_quantiles(biomarkers,
            age_reference_biomarkers, conditions)
        age_reference_qfi = pl.nanmean(age_reference_quantiles, axis = 1)

        # Using the age binned FI for FI-X
        mid_points, fi_clin_binned = bin_x_by_y(fi_clin, age_reference_qfi,
                                                                    fi_bins)
        means = [pl.average(f) for f in fi_clin_binned]
        size_mask = pl.asarray([len(f) > min_count for f in fi_clin_binned])
        means = pl.asarray(means)[size_mask]
        errors = [pl.std(f)/pl.sqrt(len(f)) for f in fi_clin_binned]
        errors = pl.asarray(errors)[size_mask]

        pl.figure(figsize = (8,6))
        pl.errorbar(mid_points[size_mask], means, yerr = errors,
            fmt = 'C0o', capsize = 3, ms = 10)
        #    label = 'Age Reference, C-index: {:.3f}'.format(c))

        pl.xlabel('QFI', fontsize = fs)
        pl.ylabel('FI-Clin', fontsize = fs)
        #pl.legend()
        pl.savefig(plots_dir + '{0}FIClinVsQFI.pdf'.format(data_set))
 
    plot_fi_vs_age = 0
    if plot_fi_vs_age:
        reference_ages = [80,85]
        min_age = min(ages) - min(ages)%bin_width
        max_age = max(ages)
        age_bins = pl.arange(min_age, max_age + bin_width, bin_width)

        age_mask = (ages >= reference_ages[0]) & (
                    ages < reference_ages[1])
        age_reference_biomarkers = biomarkers[age_mask,:]
        age_reference_quantiles = projected_quantiles(biomarkers,
            age_reference_biomarkers, conditions)
            
        qfi = pl.nanmean(age_reference_quantiles, axis = 1)

        pl.figure(figsize = (8,6))
        mid_points, qfi_age_binned = bin_x_by_y(qfi, ages, age_bins)
        qfi_size_mask = [len(a) > min_count for a in qfi_age_binned]
        qfi_means = [pl.average(f) for i, f in enumerate(qfi_age_binned)
             if qfi_size_mask[i]]
        qfi_errors = [pl.std(f)/(pl.sqrt(len(f))) for i, f in
             enumerate(qfi_age_binned) if qfi_size_mask[i]]
        pl.errorbar(mid_points[qfi_size_mask], qfi_means, yerr = qfi_errors,
            fmt = 'C0o', label = 'QFI', capsize = 3, ms = 10)
       
        mid_points, fi_clin_age_binned = bin_x_by_y(fi_clin, ages, age_bins)
        clin_size_mask = [len(a) > min_count for a in fi_clin_age_binned]
        fi_clin_means = [pl.average(f) for i, f in
            enumerate(fi_clin_age_binned) if clin_size_mask[i]]
        fi_clin_errors = [pl.std(f)/(pl.sqrt(len(f))) for i, f in 
            enumerate(fi_clin_age_binned) if clin_size_mask[i]]
        

        pl.errorbar(mid_points[clin_size_mask], fi_clin_means,
            yerr = fi_clin_errors,
            fmt = 'ks', label = 'FI-Clin', capsize = 3, ms = 8)
        pl.xlabel('Age (years)', fontsize = fs)
        pl.ylabel('FI', fontsize = fs)
        pl.legend(fontsize = fs, frameon = False)
        pl.savefig(plots_dir + '{0}FIvsAge{1}.pdf'.format(data_set, date))

    # Figure 4: ELSA-Specific Plotting Stuff: Diagnoses etc.
    diagnoses = 0
    if "ELSA" not in data_set and diagnoses:
        print("Select ELSA Data to plot Diagnosis data")
        diagnoses = 0
    if diagnoses:

        ##### FIGURE 4C: TOTAL DIAGNOSES UP TO THIS WAVE
        reference_ages = [80,85]
        matched_ages = matched_data_df['age'].values
        age_mask = (matched_ages >= reference_ages[0]) & (
                    matched_ages < reference_ages[1])
        age_reference_biomarkers = matched_biomarkers[age_mask,:]
        age_reference_quantiles = projected_quantiles(matched_biomarkers,
            age_reference_biomarkers, conditions)
        age_reference_qfi = pl.nanmean(age_reference_quantiles, axis = 1)

        # counting up the diagnoses
        total_diagnoses = matched_num_diagnoses_df['wave0'].values
        if '4' in data_set:
            total_diagnoses += matched_num_diagnoses_df['wave2'].values

        bin_width = 0.05
        qfi_bins = pl.arange(0, 1 + bin_width, bin_width)
        mid_points, qfi_binned_diagnoses_counts = bin_x_by_y(total_diagnoses,
                                            age_reference_qfi, qfi_bins)

        fig, ax = pl.subplots(1,1,figsize = (8,6))

        # have to select bins with enough people in them (min_count)
        mid_points = pl.around(mid_points, decimals = 3)
        mid_points = [mp for i, mp in enumerate(mid_points) if
            len(qfi_binned_diagnoses_counts[i]) > min_count]
        tick_positions = mid_points + [mid_points[-1] + bin_width]
        tick_positions = [pos - bin_width/2 for pos in tick_positions]
        tick_labels = ["{:.2f}".format(pos) for pos in tick_positions]
        qfi_binned_diagnoses_counts = [count for count in
            qfi_binned_diagnoses_counts  if len(count) > min_count]
        # plot on box and whisker
        pl.boxplot(qfi_binned_diagnoses_counts, showmeans = True,
            positions = mid_points, whis = [1,99],
            meanline = True, widths = 0.025,
            meanprops = dict(linestyle = '-', color = 'k', alpha = 1,
                linewidth = 1.7, label = 'Mean'),
            medianprops = dict(linestyle = '--', color = 'k', alpha = 1,
                linewidth = 1.4, label = 'Median'))
        h, l = format_legend(ax)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.legend(h, l, frameon = False, fontsize=fs,
            loc = 'upper left')
        ax.set_xlim(min(mid_points) - 0.025, max(mid_points) + 0.025)
        pl.ylabel('Total Number of Diagnoses', fontsize = fs)
        pl.xlabel('QFI', fontsize = fs)
        pl.savefig(plots_dir + '{0}TotalDiagnoses.pdf'.format(
                                                        data_set))


        #### Figure 4d: Proportion with 1 or more new diagnoses in following
        #### wave
        fig, ax = pl.subplots(1,1,figsize = (8,6))
        # Grab wave of first diagnosis data
        wave_of_first_diagnosis = matched_diagnoses_waves.values
        followup_wave = wave + 1
        diagnoses_at_followup = (wave_of_first_diagnosis == followup_wave)
        num_new_diagnoses = pl.sum(diagnoses_at_followup, axis = 1)
        mid_points, qfi_binned_diagnoses = bin_x_by_y(num_new_diagnoses,
                                                age_reference_qfi, qfi_bins)
        mid_points = pl.around(mid_points, decimals = 3)
        mid_points = [mp for i, mp in enumerate(mid_points) if
            len(qfi_binned_diagnoses[i]) > min_count]
        qfi_binned_diagnoses = [d for d in qfi_binned_diagnoses
                                        if len(d) > min_count]

        proportions = [pl.average(d > 0.5) for d in qfi_binned_diagnoses]
        errors = [pl.sqrt(p*(1-p)/len(qfi_binned_diagnoses[i])) for i, p in
            enumerate(proportions)]
        ax.errorbar(mid_points, proportions, yerr = errors,
            fmt = 'ko', capsize = 3)
        #ax.legend(h, l, frameon = False, fontsize=fs,
        #    loc = 'upper left')
        ax.set_xlim(min(mid_points) - 0.025,
            max(mid_points) + 0.025)

        pl.ylabel('Proportion with New Diagnoses', fontsize = fs)
        #pl.legend()
        pl.xlabel('QFI', fontsize = fs)
        #pl.title('{0}, {1}'.format(wave_id, date))
        pl.savefig(plots_dir + '{}NewDiagnoses.pdf'.format(data_set))



    # Figure 5: Limits of QFI with changing age reference
    # No Published FI-Lab to compare for ELSA data
    qfi_limits = 0
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
    age_paired_prediction = 1
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
        labels.append('Biomarkers + \nAge LogReg')


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


    # Figure 7: Example of sex-controlled quantiles
    # a tad wobbly based on picking the biomarker
    sex_specific_example = 0
    if sex_specific_example:
        
        # Start with an example figure
        sex_labels = ['Male', 'Female']
        sex_masks = [(sex == 1), (sex == 2)]

        values = [0, 0.2, 0.85, 1]
        colors = get_colours_from_cmap(values)

        sex_colours = ['C0', colors[2]]
        sex_markers = ['o', 's']
        
        quantiles = projected_quantiles(biomarkers, biomarkers, conditions)
        
        example_biomarker_index = 9
        example_biomarker_label = biomarker_labels[example_biomarker_index]
        example_biomarker_condition = conditions[example_biomarker_index]

        # extending the biomarker label into something legible
        use_extended_label = 1
        if use_extended_label:
            full_biomarker_label = "Mean Corpuscular Volume"
        else:
            full_biomarker_label = example_biomarker_label
        
        example_biomarker = biomarkers[:, example_biomarker_index]
        example_biomarker_quantiles = quantiles[:, example_biomarker_index]
 
        combined_sex_specific_fi = []
        combined_sex_specific_mortality = []
        fi_shifts = [] 
        auc_shifts = []
        pl.figure(figsize = (8, 6))
        for i, label in enumerate(sex_labels):
            mask = sex_masks[i]
            sex_specific_biomarkers = biomarkers[mask,:]
            non_specific_quantiles = quantiles[mask,:]
            non_specific_fi = pl.nanmean(non_specific_quantiles, axis = 1)
            sex_specific_quantiles = projected_quantiles(
                sex_specific_biomarkers, sex_specific_biomarkers, conditions)
            sex_specific_fi = pl.nanmean(sex_specific_quantiles, axis = 1)
            combined_sex_specific_fi += (list(sex_specific_fi))
            combined_sex_specific_mortality += (list(mortality[mask]))
            
            fi_shifts.append(sex_specific_fi - non_specific_fi)
            auc_shifts.append(roc_auc_score(mortality[mask], sex_specific_fi)
                - roc_auc_score(mortality[mask], non_specific_fi))
            non_specific_example_biomarker_quantiles = (
                example_biomarker_quantiles[mask])
            sex_specific_example_biomarker_quantiles = rank_normalize(
                        example_biomarker[mask], example_biomarker_condition)
            pl.plot(example_biomarker[mask],
                non_specific_example_biomarker_quantiles, ms = 6 - 3*i,
                ls = '', color = sex_colours[i], mfc = 'none',
                marker = sex_markers[i])

            pl.plot(example_biomarker[mask],
                sex_specific_example_biomarker_quantiles, ms = 7,
                ls = '', color = sex_colours[i], marker = sex_markers[i])

            #### Dummy plots to make the legend look better
            pl.plot(pl.nan, pl.nan, ls = '',
                color = sex_colours[i], mfc = 'none', marker = sex_markers[i],
                label = '{}, Non-Adjusted'.format(sex_labels[i]), ms = 6)
            pl.plot(pl.nan, pl.nan, ls = '', ms = 10,
                color = sex_colours[i], marker = sex_markers[i],
                label = '{}, Sex-Adjusted'.format(sex_labels[i]))


        pl.legend(fontsize = fs, frameon = False)
        pl.xlabel(full_biomarker_label, fontsize = fs)
        pl.ylabel('x', fontsize = fs)
        pl.savefig(plots_dir + '{0}SexDifference{1}QuantileExample.pdf'.format(
                data_set, example_biomarker_label))


        # Plotting the shift in FI
        pl.figure(figsize = (8, 6))
        female_shift = fi_shifts[1]
        male_shift = fi_shifts[0]
        bw = 0.1/20
        bins = pl.arange(pl.amin(female_shift), pl.amax(male_shift) + bw, bw)
        pl.hist(female_shift, bins, color = sex_colours[1],
                label = sex_labels[1])
        pl.hist(male_shift, bins, color = sex_colours[0],
                label = sex_labels[0])
        counts = pl.hist(female_shift, bins, edgecolor = sex_colours[1],
            facecolor = 'none',)[0]
        #    hatch = 'none')
        pl.legend(loc = 'upper right', fontsize = fs, frameon = False)
        pl.xlabel('Shift in FI', fontsize = fs)
        pl.ylabel('Count', fontsize = fs)
        pl.ylim(0, 1.1*max(counts))
        pl.savefig(plots_dir + '{}SexDifferenceFIShift.pdf'.format(
            data_set))


            

        # Distributions example
        n_bins = 20
        biomarker_range = -(pl.nanmin(example_biomarker) -
            pl.nanmax(example_biomarker))
        bw = biomarker_range/n_bins
        bins = pl.arange(pl.nanmin(example_biomarker),
            pl.nanmax(example_biomarker) + bw, bw)
         
        pl.figure(figsize = (8,6))

        example_biomarker_male = example_biomarker[sex_masks[0]]
        example_biomarker_female = example_biomarker[sex_masks[1]]
        pl.hist(example_biomarker_female, bins, color = sex_colours[1],
            label = sex_labels[1])
        pl.hist(example_biomarker_male, bins, color = sex_colours[0],
            label = sex_labels[0])
        counts = pl.hist(example_biomarker_female, bins,
            edgecolor = sex_colours[1], facecolor = 'none',)[0]
        pl.legend(loc = 'upper right', fontsize = fs, frameon = False)
        pl.xlabel(full_biomarker_label, fontsize = fs)
        pl.ylabel('Count', fontsize = fs)
        pl.ylim(0, 1.1*max(counts))
        pl.savefig(plots_dir + '{0}SexDifference{1}Distribution.pdf'.format(
            data_set, example_biomarker_label))


    # Figure 7: sex-controlled QFI vs Age and QFI
    plot_stratified_qfi = 0
    if plot_stratified_qfi:
        sex_labels = ['Male', 'Female']
        sex_masks = [(sex == 1), (sex == 2)]
        vorders = [10, 1]

        values = [0, 0.2, 0.85, 1]
        colors = get_colours_from_cmap(values)

        sex_colours = ['C0', colors[2]]
        sex_markers = ['o', 's']

        age_mask = (ages >= 80) & (ages < 85)
        quantiles_80 = projected_quantiles(biomarkers, biomarkers[age_mask,:],
                                                                conditions)
        qfi_80 = pl.nanmean(quantiles_80, axis = 1)
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
            non_adjusted_qfi = qfi_80[sex_mask]
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

            non_adjusted_qfi = qfi_80[sex_mask]
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


        pl.legend(fontsize = fs*0.7)
        pl.xticks(fontsize = fs*0.8)
        pl.yticks(fontsize = fs*0.8)
        pl.xlabel('Age (years)', fontsize = fs)
        pl.ylabel('QFI', fontsize = fs)
        pl.savefig(plots_dir + '{}SexAdjustedQFIVsAge.pdf'.format(data_set))

        ##### Ugly bit of work here, some gory details made it worth doing this
        # whole shabang over again (in my head at least)

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

            length_mask = [len(a) > min_count for a in binned_adjusted_qfi]
            adjusted_means = [pl.average(a) for a in binned_adjusted_qfi
                if len(a) > min_count]
            adjusted_errors = [pl.std(a)/pl.sqrt(len(a)) for a in
                binned_adjusted_qfi if len(a) > min_count]

            pl.errorbar(mid_points[length_mask], adjusted_means,
                yerr = adjusted_errors,
                c = sex_colours[i], marker = sex_markers[i], ls = 'none',
                capsize = 3, label = label + " Adjusted", ms = 7 + 2*i,
                zorder = vorders[i])


            non_adjusted_qfi = qfi_80[sex_mask]
            mid_points, binned_qfi = bin_x_by_y(fi_clin[sex_mask],
                non_adjusted_qfi, qfi_bins)
            length_mask = [len(a) > min_count for a in binned_qfi]

            non_adjusted_means = [pl.average(a) for a in
                binned_qfi if len(a) > min_count]
            non_adjusted_errors = [pl.std(a)/pl.sqrt(len(a)) for a in
                binned_qfi if len(a) > min_count]
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
        pl.savefig(plots_dir + '{0}SexAdjustedFIClinVsQFI.pdf'.format(
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
        quantiles = projected_quantiles(biomarkers, biomarkers[age_mask,:],
                                                                conditions)
        qfi = pl.nanmean(quantiles, axis = 1)
        aucs.append(roc_auc_score(mortality, qfi))
        # adjust for sex
        mortality_list = []
        sex_adjusted_qfi_list  = []
        fully_adjusted_qfi_list = []
        sex_specific_conditions = []
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
            age_sex_qfi = age_paired_qfi(sex_biomarkers, ages[sex_mask],
                                                conditions, bin_width = 5)
            fully_adjusted_qfi_list.append(age_sex_qfi)
            aucs.append(roc_auc_score(mortality[sex_mask], age_sex_qfi))

        aucs.insert(1, roc_auc_score(pl.concatenate(mortality_list),
                                pl.concatenate(sex_adjusted_qfi_list)))
        aucs.insert(2, roc_auc_score(pl.concatenate(mortality_list),
                                pl.concatenate(fully_adjusted_qfi_list)))

        values = [0, 0.2, 0.85, 1]
        colors = get_colours_from_cmap(values)
        

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
            sampled_mortality = split_data[2]
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
                sex_mortality = sampled_mortality[sex_mask]
                mortalities.append(sex_mortality)
                # Sex + 80-85 year old reference for QFI-80
                both_mask = sex_mask & age_mask
                sex_biomarkers = sampled_biomarkers[sex_mask,:]
                sex_reference_biomarkers = sampled_biomarkers[both_mask, :]
                sex_adjusted_quantiles = projected_quantiles(sex_biomarkers,
                    sex_reference_biomarkers, conditions)
                sex_adjusted_qfi = pl.nanmean(sex_adjusted_quantiles, axis = 1)
                sex_adjusted_qfis.append(sex_adjusted_qfi)
                resampled_aucs[4 + 3*i].append(roc_auc_score(sex_mortality,
                    sex_adjusted_qfi))
                # Non-Adjusted qfi (just grabbed from full qfi quantilesribution)    
                non_adjusted_qfi = qfi[sex_mask]
                resampled_aucs[3 + 3*i].append(roc_auc_score(sex_mortality,
                    non_adjusted_qfi))
                # Age and sex paired QFIs:
                age_sex_qfi = age_paired_qfi(sex_biomarkers,
                    sampled_age[sex_mask], conditions, bin_width = 5)
                fully_adjusted_qfis.append(age_sex_qfi)
                resampled_aucs[5+3*i].append(roc_auc_score(sex_mortality,
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
        ax.legend(h, l, frameon = False, loc = 'lower left', fontsize = 0.8*fs)
        ax.grid(alpha = 0.3)
        pl.savefig(plots_dir + '{}SexPrediction.pdf'.format(data_set))

    ### SUPPLEMENTAL FIGURES        
    # Demographics: Age Distributions  With mortality also
    plot_age_distribution = 0
    if plot_age_distribution:
        bin_width = 5
        age_bins = pl.arange(20, 110, bin_width)
        mortality_counts = [
            pl.sum(mortality[(ages >= low_age) & (ages < age_bins[i+1])])
            for i, low_age in enumerate(age_bins[:-1])]
        pl.figure(figsize = (8,6))
        counts, bins = pl.hist(ages, age_bins,
            label = "Age Distribution\nTotal Number:{}".format(
                                                    len(mortality)))[:2]
        bins = bins[:-1]
        pl.xlabel('Age (years)', fontsize = fs)
        pl.ylabel('Count', fontsize = fs)
        pl.bar(bins, mortality_counts, width = bin_width, align = 'edge', 
            color = 'C1',
            label = "Mortality Distribution\nTotal Number:{}".format(
                                                            pl.sum(mortality)))
        filled_bins = bins[counts > 0]
        pl.xlim(min(filled_bins) - 1, max(filled_bins) + 6)
        pl.legend(fontsize = fs)
        pl.savefig(plots_dir + '{}AgeDistribution.pdf'.format(data_set))

    # Details of age averaged predictions plots
    plot_age_stratified_auc = 0
    if plot_age_stratified_auc:
        reference_ages = [80,85]
        age_mask = (ages >= reference_ages[0]) & (
                    ages < reference_ages[1])
        age_reference_biomarkers = biomarkers[age_mask,:]
        age_reference_quantiles = projected_quantiles(biomarkers,
            age_reference_biomarkers, conditions)
            
        age_reference_qfi = pl.nanmean(age_reference_quantiles, axis = 1)
        age_reference_auc = roc_auc_score(mortality, age_reference_qfi)
        age_paired_qfi = age_paired_qfi(biomarkers, ages, conditions)
        
        
        age_paired_averaged_auc, age_paired_aucs, mid_points = (
                    age_averaged_prediction(age_paired_qfi, mortality, ages,
                                                            ret_all = True))
        age_paired_auc = roc_auc_score(mortality, age_paired_qfi) 
        age_reference_averaged_auc = age_averaged_prediction(
                                            age_reference_qfi, mortality, ages)
        pl.plot(mid_points, age_paired_aucs, 'C0o',
            label = 'Age-Restricted QFI')
        pl.axhline(age_paired_auc, ls = ":",
            label = 'Age-Paired QFI')
        pl.axhline(age_reference_auc, ls = "-",
            label = 'No Age Control')
        pl.xlabel('Age (years)', fontsize = fs)
        pl.ylabel('AUC', fontsize = fs)
        pl.legend()
        pl.savefig(plots_dir +
            '{0}AgeStratifiedPrediction.pdf'.format(data_set))
 
            

    

    pl.show()
