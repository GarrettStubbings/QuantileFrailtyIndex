"""

Useful functions for Quantile Frailty Index code contained here

"""
import pylab as pl
import numpy
import pandas as pd
import operator
import numpy as np
import datetime
from scipy import stats
import lifelines
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.model_selection import *
import matplotlib as mpl
import matplotlib.cm as cmx
from matplotlib.ticker import AutoMinorLocator

pl.close('all')
pd.set_option('float_format', '{:.4f}'.format)

date = datetime.date.today().strftime('%e%b%y')
date = date.strip(' ')

fs = 16
bin_width = 5
min_count = 20

def format_legend(ax):
    """formats the legend of a subplots thingy to chuck away duplocate labels.
    returns handles, lables"""
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for i in range(len(labels)):
        if labels[i] not in unique_labels:
            unique_labels.append(labels[i])
            unique_handles.append(handles[i])
    return unique_handles, unique_labels

def get_colours_from_cmap(values, cmap = 'viridis'):
    """
    Function for getting line / point colours from colormaps
    Better colorschemes to show sequences.
    e.g. when the legend is something like (a = 1, a = 2, a = 3, etc)
    """

    cm = mpl.cm.viridis(pl.linspace(0, 1, 100))
    cm = mpl.colors.ListedColormap(cm)
    c_norm = pl.Normalize(vmin=0, vmax=values[-1])
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap = cm)
    colours = [scalar_map.to_rgba(v) for v in values]

    return colours

def list_logic(list_data, logic):
    """
    function to do advanced array indexing on lists in 1 line.
    Used in list comprehension heavy sections 
    """
    data = pl.asarray(list_data)
    return list(data[logic])

def resolution(quantiles, N = -1):
    """
    Scales the resolution of the data (dichotomize vs tertile vs quantile)
    risk category goes as floor(x * m)/(m-1)
    """
    if N == -1 or N == 'all' or N == 'full':
        return quantiles

    else:
        scaled_quantiles = N *quantiles 
        floor_quantiles = pl.full_like(quantiles, pl.nan)
        floor_quantiles[~pl.isnan(quantiles)] = scaled_quantiles[
                    ~pl.isnan(quantiles)].astype(int)
        coarse_quantiles = floor_quantiles / (N-1)

    return coarse_quantiles

def projected_quantiles(data, reference_data, conds):
    """
    Calculate the quantiles that the data would land in in the reference
    data
    can have data == reference data, for e.g. qfi-all
    requires conditions so that quantiles are ascending with risk
    """

    projected_dist = pl.zeros_like(data)

    for i in range(pl.shape(data)[1]):
        data_col = data[:,i]
        dist_col = pl.full(len(data_col), pl.nan)
        if pl.sum(pl.isnan(data_col)) == len(data_col):
            projected_dist[:, i] = dist_col
            continue
        reference_col = reference_data[:,i]

        reference_col = reference_col[~pl.isnan(reference_col)]
        if len(reference_col) == 0:
            projected_dist[:, i] = dist_col
            continue

        sorted_reference_col = pl.sort(reference_col)

        if conds[i] == operator.gt:
            side = 'left'
            dist_col[~pl.isnan(data_col)] = pl.searchsorted(
                sorted_reference_col, data_col[~pl.isnan(data_col)],
                side = side)


        if conds[i] == operator.lt:
            side = 'right'
         
            dist_col[~pl.isnan(data_col)] = pl.searchsorted(
                -sorted_reference_col[::-1],
                -data_col[~pl.isnan(data_col)], side = side)

        dist_col /= len(reference_col[~pl.isnan(reference_col)])

        projected_dist[:, i] = dist_col

    return projected_dist

def age_averaged_prediction(fi, mort, ages, measure = 'auc', bin_years = 5,
    weighted = True):
    """ weighted (or not) average of predictive value over age groups """
    bins = pl.arange(min(ages), max(ages) + bin_years, bin_years)
    fi_binned = bin_x_by_y(fi, ages, bins)[1]
    mort_binned = bin_x_by_y(mort, ages, bins)[1]
    valid_bins = [(pl.average(m) != 1) & (pl.average(m) != 0) for m in
                            mort_binned]
    fi_binned = [f for i, f in enumerate(fi_binned) if valid_bins[i]]
    mort_binned = [m for i, m in enumerate(mort_binned) if valid_bins[i]]
    total_considered_events = 0
    for m in mort_binned:
        total_considered_events += len(m)
    
    weights = [len(m)/total_considered_events for i,m in 
                                enumerate(mort_binned) if len(m) > 0]
    
    aucs = [roc_auc_score(m, fi_binned[i]) for i, m in
        enumerate(mort_binned) if len(m) > 0]
    if weighted:
        return pl.average(aucs, weights = weights)
    else:
        return pl.average(aucs)

def bin_x_by_y(x, y, bins):
    """Bin some data by the other data
    return middle point of bins, data"""
    binned_data = []
    means = []
    errors = []
    mid_points = (bins[:-1] + bins[1:])/2

    if len(pl.shape(x)) == 1:
        for i in range(1, len(bins)):
            data = x[(y >= bins[i-1]) & (y < bins[i])]
            binned_data.append(data)

    else:
        for i in range(1, len(bins)):
            data = x[(y >= bins[i-1]) & (y < bins[i]), :]
            binned_data.append(data)
        
    return mid_points, binned_data

def spearman_age_conditions(data, ages, condition_age):
    """this function uses the spearman monotonicity thingy to calculate the
    conditions appropriate for all deficits in a dataset.
    Takes a real value dataset, ages as input.
    Returns age conditions."""
    # Dont include individuals below condition age in calculations
    data = data[(ages >= condition_age), :]
    ages = ages[ages >= condition_age]
    spearman_age_correlations = []
    # go through all deficits
    for d in range(pl.shape(data)[1]):

        col = data[:,d]
        if sum(~pl.isnan(col)) == 0:
            print("column {} has al nans".format(d))
            spearman_age_correlations.append(-2)
            continue
        # calculate the p values and correlations for the spearman stuff
        correlation, p_value = stats.spearmanr(col, ages, nan_policy = 'omit')
        spearman_age_correlations.append(correlation)

    spearman_age_conds = [operator.lt if c < 0 else operator.gt for c in 
        spearman_age_correlations]

    return spearman_age_conds

def age_paired_qfi(biomarkers, ages, conds, bin_width = 5):
    age_bins =  pl.arange(min(ages) - min(ages)%bin_width,
        max(ages) + 2*bin_width, bin_width)
    qfi_age_paired = pl.full(len(ages), pl.nan)
    for i, low_age in enumerate(age_bins[:-1]):
        high_age = age_bins[i+1]
        
        age_mask = (ages >= low_age) & (ages < high_age)

        reference_biomarkers = pl.copy(biomarkers[age_mask,:])
        risk_quantiles = projected_quantiles(reference_biomarkers,
            reference_biomarkers, conds)
        qfi_age_paired[age_mask] = pl.nanmean(risk_quantiles, axis = 1)
    return qfi_age_paired

def fi_q_bootstrap(data, mortality, conds, n_samp = 100, n_quantiles = 'all',
    measure = 'auc'):
    """ Bootstrap fi_q with the option to degrade the number of quantiles """
    data = pl.column_stack((data, mortality))
    size = len(mortality)
    size = int(size/2)
    values = []
    for n in range(n_samp):
        pl.shuffle(data)
        mort = pl.copy(data[:,-1])
        
        sample_data = pl.copy(data[:size, :-1])
        sample_mort = pl.copy(mort[:size])

        sample_dist = projected_quantiles(sample_data, sample_data, conds)

        coarse_dist = resolution(sample_dist, n_quantiles)

        sample_fi = pl.nanmean(coarse_dist, axis = 1)


        if measure == 'auc':
            values.append(roc_auc_score(sample_mort, sample_fi))
        else:
            values.append(FrailtyInfo(sample_fi, mort))

    return values

def qfi_reference_bootstrap(data, reference_data, mort, conds, n_samp = 100,
    measure = 'auc', n_quantiles = -1):
    """ Bootstrap resample the QFI using a reference population """
    #print("Data shape:", data.shape, ". Mortality shape:", mort.shape)
    data = pl.column_stack([data, mort])
    size = len(mort)
    size = int(size/2)
    values = []
    for n in range(n_samp):
        pl.shuffle(data)
        mort = pl.copy(data[:,-1])
        
        sample_data = pl.copy(data[:size, :-1])
        sample_mort = pl.copy(mort[:size])

        sample_dist = projected_quantiles(sample_data, reference_data, conds)
        coarse_quantiles = resolution(sample_dist, n_quantiles)

        sample_fi = pl.nanmean(coarse_quantiles, axis = 1)

        if measure == 'auc':
            values.append(roc_auc_score(sample_mort, sample_fi))
        else:
            values.append(FrailtyInfo(sample_fi, mort))

    return values

def reference_population(data_dict, reference_range = [80, 85],
    reference_type = 'age', reference_ages = None):
    """ select a reference population """
    i = 0
    if type(data_dict) == np.ndarray:
        print('Array Data Inputted')
        biomarkers = data_dict
        reference_values = reference_ages
        mask = ((reference_values >= reference_range[0]) &
            (reference_values < reference_range[1]))
        masked_biomarkers = biomarkers[mask, :]
        combined_biomarkers = masked_biomarkers

    else:
        for k, v in data_dict.items():
            biomarkers = v['biomarkers']
            
            reference_values = v[reference_type]

            mask = ((reference_values >= reference_range[0]) &
                (reference_values < reference_range[1]))
            masked_biomarkers = biomarkers[mask, :]
            if i == 0:
                combined_biomarkers = masked_biomarkers
                i += 1
            else:
                combined_biomarkers = pl.vstack([combined_biomarkers,
                    masked_biomarkers])

    return combined_biomarkers
 
def cutpoint_calculator(data, cond, frac):
    """This function takes a fraction of the population (frac) and returns
    the value of the cutpoint for each deficit"""
    cutpoints_list = []
    # go through each deficit
    for i in range(pl.shape(data)[1]):
        # copy th column
        col = pl.copy(data[:,i])
        col = col[~pl.isnan(col)]
        col_sorted = pl.sort(col)
        if cond[i] == operator.lt:
            col_sorted = col_sorted[::-1]
        if cond[i] != operator.eq:
            cut_index = int(frac*len(col_sorted))
            if cut_index >= len(col_sorted):
                if cond[i] == operator.lt:
                    cut_value = col_sorted[-1] - 1
                else:
                    cut_value = col_sorted[-1] + 1
            elif cut_index == 0:
                if cond[i] == operator.lt:
                    cut_value = col_sorted[0] + 1
                else:
                    cut_value = col_sorted[0] - 1
            else:
                cut_value = col_sorted[cut_index]
        else:
            cut_value = 1
        cutpoints_list.append(cut_value)
    return cutpoints_list

def cutpoint_fi(data, cutoffs, cond):
    """Takes continuous valued deficit information and first binarises it
    then produces a frailty index out of it, the outputed frailty index
    is a 1D array (FI for each person)"""
    if type(cutoffs) != np.ndarray and type(cutoffs) != list:
        cutoffs = pl.full(len(cond), cutoffs)
    binary = pl.copy(data)
    # binarize each column by the deficit cutoff value
    for i in range(pl.shape(data)[1]):
        # copy th column
        col = pl.copy(data[:,i])
        # binarize the non nan values (leave the nans in)
        col[~pl.isnan(col)] = cond[i](col[~pl.isnan(col)],
            cutoffs[i]).astype(int)
        binary[:,i] = col
    frailty = []
    # take each row(person) and average the existing binary data for FI
    for i in range(pl.shape(data)[0]):
        row = pl.copy(binary[i,:])
        row = row[~pl.isnan(row)]
        fi = pl.average(row)
        frailty.append(fi)
    return pl.asarray(frailty)

# In stead of dataframe, just use data and a mort col
def fi_gcp_cross_validation(data, mort, cutpoints, cond, cutpoint_type,
        metric = 'Information', samples = 5,
        subset_count = 2, mortality = 'mortality', resolution = 100,
        use_max = False, non_def_cols = 0):
    """takes an array of continuous values health data, the number of
    subsets to divide it into, the cutpoints: either an array of continuous
    valued health cutpoints or a fraction of the distribution based on
    the cutpoint type, either 'value or 'fraction' also takes a number of
    samples, which is the number of times to randomly shuffle the data set and
    perform the calculation. Mortality for FI_info is the morality column"""
    samples = int(samples/2)   
    # total number of deficits
    n_def = pl.shape(data)[1] - non_def_cols
    # attach the mortality column to the deficits data for shuffling
    data = pl.column_stack((data, mort))
    if cutpoint_type == 'fraction':
        cutpoints = pl.full(n_def, cutpoints)
        cut_frac = cutpoints[0]
    subset_len = int(pl.shape(data)[0]/subset_count)
    predictions = []
    # randomly sort the data n times, note that it returns 2n datapoints
    for n in range(samples):
        pl.shuffle(data)
        mort = pl.copy(data[:,-1])
        # k will determine which subset to be tested on(division of data)
        for k in range(subset_count):
            # subdivide the data and mortality into the training and test sets
            test = pl.copy(data[subset_len*k:subset_len*(k+1),: n_def])
            train = pl.delete(data,pl.arange(subset_len*k,subset_len*(k+1)),
                axis = 0)[:,:n_def]
            test_mort = pl.copy(mort[subset_len*k:subset_len*(k+1)])
            train_mort = pl.delete(mort, pl.arange(subset_len*k, subset_len*(
                k+1)))
            if cutpoint_type == 'max':
                cutpoints = max_info_cut(train, cond, train_mort)
                if n == 0:
                    print('Resolution:{0}, use_max:{1}'.format(resolution,
                        use_max))
            elif cutpoint_type == 'max fraction':
                cutpoints = pl.empty(n_def)
                cut_fractions = max_info_cut(train, cond, train_mort,
                    resolution = resolution, use_max = use_max)[2]
                if n == 0:
                    print('Resolution:{0}, use_max:{1}'.format(resolution,
                        use_max))
                # go over each deficit (last 4 cols arent deficits)
                for i in range(n_def):
                    test_col = pl.copy(test[:,i])
                    train_col = pl.copy(train[:,i])
                    train_sorted = pl.sort(train_col[~pl.isnan(train_col)])
                    if cond[i] == operator.lt:
                        train_sorted = train_sorted[::-1]
                    cut_index = int(cut_fractions[i]*len(train_sorted))
                    # append the continuous valued deficit
                    cutpoints[i] = train_sorted[cut_index]
                    # use the classic for binary cutpoints
                    if cond[i] == operator.eq:
                        cutpoints[i] = 1
            else:
                # go over each deficit (last 4 cols arent deficits)
                for i in range(n_def):
                    test_col = pl.copy(test[:,i])
                    train_col = pl.copy(train[:,i])
                    
                    # if not using predetermined cutpoints
                    if cutpoint_type == 'fraction':
                        train_sorted = pl.sort(train_col[~pl.isnan(train_col)])
                        if cond[i] == operator.lt:
                            train_sorted = train_sorted[::-1]
                        cut_index = int(cut_frac*len(train_sorted))
                        # append the continuous valued deficit
                        cutpoints[i] = train_sorted[cut_index]
                    # use the midpoint for binary cutpoints
                    if cond[i] == operator.eq:
                        cutpoints[i] = 1
            test_frailty = cutpoint_fi(test, cutpoints, cond)
            if metric == 'Information':
                prediction = FrailtyInfo(test_frailty, test_mort)
            else:
                prediction = roc_auc_score(test_mort, test_frailty)
            predictions.append(prediction)
    return predictions


