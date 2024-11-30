from used_packages import *


#%%

def clean_title(title):
    return title.replace('\n', '-')


#%%

#replace character inside DF column names (rename columns)

def replace_col_names(data, replace_char, with_char):
    def replace_slash(column_name):
        return column_name.replace(replace_char, with_char)
    data_temp = data.copy()
    # Rename columns
    new_columns = {column: replace_slash(column) for column in data_temp.columns}
    data_temp.rename(columns=new_columns, inplace=True)

    return data_temp

#%%

def one_hot_encode_pandas(df, response, except_col=[], include_col=[], dtype='float32', drop_first=True):

    # Separate numerical, categorical, and boolean columns, and response column
    numerical_columns = df.select_dtypes(include=['number']).columns
    boolean_columns = df.select_dtypes(include=['bool']).columns

    if except_col:
        categorical_columns = [col for col in df.columns if col not in numerical_columns and col not in boolean_columns
                               and col not in except_col or col in include_col]
    else:
        categorical_columns = [col for col in df.columns if col not in numerical_columns and col not in boolean_columns]

    # Create the design matrix
    df_encoded = pd.get_dummies(df, columns=categorical_columns , drop_first=drop_first, dtype=dtype)

    N = []
    for num, label in enumerate(df.columns.tolist()):
        for dum_label in df_encoded.columns.tolist():
            if label == dum_label[:len(label)]:
                N.append(dum_label)

    df_rearranged = df_encoded[N]

    df = df_rearranged

    df = replace_col_names(df, ' ', '_')

    # Extract response and features
    y = pd.DataFrame(df[response])
    X = df.drop(columns=[response])  # Cast features to float

    return df, X, y

#%%
def assign_quantile_group(row, group, quantiles, column_name):
    col = row[column_name]
    value = group[col]

    # Initialize the group number (category) as the first category
    group_category = f"1"

    # Iterate over the quantiles and assign the group category
    for i, q in enumerate(quantiles):
        if value >= q:
            group_category = f"{i + 2}"  # Group names start from "Group 1"
        else:
            break  # Once the value is less than the quantile, stop checking

    return group_category

#%%
def determine_lasso_group(lasso_coefs: pd.Series, frac: float):
    l = lasso_coefs.copy().abs()
    # Initialize the group number (category) as the first category
    group_category = f"1"

    # Iterate over the quantiles and assign the group category
    i = 0
    val = np.inf
    groups = []
    for ind, coef in l.items():

        # The function checks if the abs. change in coef valu is
        # greater than the highest coef in the group times frac,
        # if yes, create new group label, update the value of the
        # highest coef in the group, exist the if statement and
        # assign new group label to the current coef.
        # If no, assign current group label to the current coef.

        if abs(coef - val) >= val * frac:
            val = coef
            group_category = f"{i + 1}"
            i += 1

        groups.append(group_category)


    l = l.reset_index(drop=False).drop(columns=['coef'])
    concat = pd.concat([l, pd.Series(groups, name='group')], axis=1).reset_index(drop=True)

    return concat.set_index('cat').squeeze()

#%%
def bar_plot(group, main_title=None, x_label=None, ylim=None, fig_size=None, abs=False, num_percentiles=None,yticks=True, xticks=False, save_plot=False):
    plt.figure(figsize=fig_size)
    if abs:
        group_values = np.abs(group.values)
    else:
        group_values = group.values

    plt.bar(group.index, group_values)
    plt.title(main_title)
    plt.ylim(ylim)
    plt.xlabel(x_label)

    if xticks:
        plt.xticks(rotation=90)
    else:
        plt.xticks([])

    if not yticks:
        plt.yticks([])

    # Calculate the nth percentiles for annotation positions
    num_ticks = num_percentiles  # number of percentiles
    percentiles = np.linspace(0, 1, num_ticks+1)[::-1]  # Reverse to get 100% to 0%
    percentile_indices = np.percentile(np.arange(len(group.index)), np.linspace(0, 100, num_ticks+1))

    # get vertivcal scale of plot
    y_lim_diff = np.abs(plt.gca().get_ylim()[0] - plt.gca().get_ylim()[1])


    # Add text annotations above or below the 0 line
    if num_percentiles:
        for idx, percentile in zip(percentile_indices, percentiles):
            value = group_values[int(idx)]
            if value <= 0:
                plt.text(idx, 0+0.02*y_lim_diff, f"{percentile:.2f}", ha='right', va='bottom', rotation=90)
            else:
                plt.text(idx, 0-0.02*y_lim_diff, f"{percentile:.2f}", ha='left', va='top', rotation=90)
    if save_plot:
        title_save = clean_title(main_title)
        plt.savefig(f'{title_save}.png', bbox_inches='tight', pad_inches=0.1)

    plt.show()



#%%
def sub_split(X, y, random_state=0, verbose=False):
    # fraud indecis
    i_f = np.argwhere(y == 1)[:,0]
    # non-fraud indecis
    i_nf = np.argwhere(y == 0)[:,0]

    X_fraud = X.iloc[i_f]
    y_fraud = y.iloc[i_f]

    X_non_fraud = X.iloc[i_nf]
    y_non_fraud = y.iloc[i_nf]

    # down-sampled non-fraud
    X_non_fraud_sub = X_non_fraud.sample(len(i_f), random_state=random_state)
    y_non_fraud_sub = y_non_fraud.sample(len(i_f), random_state=random_state)

    # combine down-sampled non-fraud with fraud
    X_sub = pd.concat((X_non_fraud_sub, X_fraud), axis=0)
    y_sub = pd.concat((y_non_fraud_sub, y_fraud), axis=0)

    np.random.seed(random_state)
    random.seed(random_state)

    # Train data
    i_sub = y_sub.index

    i_train, i_remaining = train_test_split(i_sub, test_size=0.2, random_state=10)
    i_temp = [i for i in X.index if i not in i_sub] # test indices that do not include class 1

    X_train = X.iloc[i_train]
    y_train = y.iloc[i_train]

    # Test data

    # indices of class 1 among the 20% split (i_remaining). These will be concat. with class 0 from outside the sub-sample.
    ind_f_remain = np.argwhere(y.iloc[i_remaining] == 1)[:,0]
    # ratio of class 1 to class 0
    frac_class = len(i_f) / (len(i_f) + len(i_nf))
    # dividing remaining split indices by frac_class to obtain no. of class 0 we need for the test set
    frac_sample = int(len(ind_f_remain) / frac_class)
    # sampling class 0 from i_temp and concatenate them with i_remaining
    i_n_remain = pd.DataFrame(i_temp).sample(frac_sample, random_state=random_state).values.flatten()
    i_test = np.concatenate((i_n_remain, i_remaining[ind_f_remain]))

    X_test = X.iloc[i_test]
    y_test = y.iloc[i_test]

    if verbose:
        # check count per class
        print('===============\nTrain Count\n===============')
        print(y_train.groupby('fraud')['fraud'].count(), '\n')
        print('===============\nTest Count\n===============')
        print(y_test.groupby('fraud')['fraud'].count())

    return X_train, y_train, X_test, y_test

#%%

def sub_split_flex(X, y, lam=0, random_state=0, verbose=False):
    # fraud indecis
    i_f = np.argwhere(y == 1)[:,0]
    # non-fraud indecis
    i_nf = np.argwhere(y == 0)[:,0]

    u = lam * (len(i_nf)) + (1-lam)*len(i_f)

    X_fraud = X.iloc[i_f]
    y_fraud = y.iloc[i_f]

    X_non_fraud = X.iloc[i_nf]
    y_non_fraud = y.iloc[i_nf]

    # down-sampled non-fraud
    X_non_fraud_sub = X_non_fraud.sample(int(u), random_state=random_state)
    y_non_fraud_sub = y_non_fraud.sample(int(u), random_state=random_state)

    # combine down-sampled non-fraud with fraud
    X_sub = pd.concat((X_non_fraud_sub, X_fraud), axis=0)
    y_sub = pd.concat((y_non_fraud_sub, y_fraud), axis=0)

    np.random.seed(random_state)
    random.seed(random_state)

    # Train data
    i_sub = y_sub.index

    i_train, i_remaining = train_test_split(i_sub, test_size=0.2, random_state=10)
    i_temp = [i for i in X.index if i not in i_sub] # test indices that do not include class 1

    X_train = X.iloc[i_train]
    y_train = y.iloc[i_train]

    # Test data - I will maintain the ratio between class 1 and 0 from the original dataset

    # indices of class 1 among the 20% split (i_remaining). These will be concat. with class 0 from outside the sub-sample.
    ind_f_remain = np.argwhere(y.iloc[i_remaining] == 1)[:,0]
    # ratio of class 1 to class 0
    frac_class = len(i_f) / (len(i_f) + len(i_nf))
    # dividing remaining split indices by frac_class to obtain no. of class 0 we need for the test set
    frac_sample = int(len(ind_f_remain) / frac_class)
    # sampling class 0 from i_temp and concatenate them with i_remaining
    i_n_remain = pd.DataFrame(i_temp).sample(frac_sample, random_state=random_state).values.flatten()
    i_test = np.concatenate((i_n_remain, i_remaining[ind_f_remain]))

    X_test = X.iloc[i_test]
    y_test = y.iloc[i_test]

    if verbose:
        i_f_new = np.argwhere(y_train == 1)[:,0]
        # non-fraud indecis
        i_nf_new = np.argwhere(y_train == 0)[:,0]
        # check count per class
        print('===============\nTrain Count\n===============')
        print(y_train.groupby('fraud')['fraud'].count(), '\n')
        print('===============\nTest Count\n===============')
        print(y_test.groupby('fraud')['fraud'].count())
        print(f'\nOriginal class 1 to class 0 ratio for the training set: {len(i_f) / len(i_nf)*100:.5f}%')
        print(f'New class 1 to class 0 ratio for the training set: {len(i_f_new) / len(i_nf_new)*100:.5f}%')

    return X_train, y_train, X_test, y_test

#%%
def glasso_groups_func(original_labels, OHE_labels):
    # List of variables after One-Hot_Encoding (long list)
    LONG = []
    # List of original variables in same order as OHE list (short list)
    SHORT = []

    # dictionary to map original labels to their group number
    label_to_group = {label: i + 1 for i, label in enumerate(original_labels)}

    for dum_label in OHE_labels:
        for label in original_labels:
            if dum_label.startswith(label):
                LONG.append(label_to_group[label])
                if label not in SHORT:
                    SHORT.append(label)
                break

    return LONG, SHORT

#%%
def classification_plots(ytrue, ypred, l_lim=0, u_lim=1, num_thresh=50, criterion='f1', title=[], legend_loc='best', save_plot=False):
    recall = []
    f1 = []
    avg_recall = []
    f1_recall_diff = []
    thresholds = np.linspace(l_lim, u_lim, num_thresh)



    for thresh in thresholds:
        y_thresh = [1 if i > thresh else 0 for i in ypred]
        cls = classification_report(ytrue, y_thresh, output_dict=True)
        recall.append(cls['1']['recall'])
        f1.append(cls['1']['f1-score'])
        avg_recall.append(cls['macro avg']['recall'])
        if f1[-1] >= 1e-1:
            f1_recall_diff.append(np.abs(recall[-1] - f1[-1]))
        else:
            f1_recall_diff.append(float(np.inf))

    if criterion is 'f1_recall_diff':
        i_max = np.argmin(f1_recall_diff)
    elif criterion is 'f1':
        i_max = np.argmax(f1)
    else:
        print('Criterion should only be "f1" or "f1_recall_diff"')



    thresh_max = thresholds[i_max]
    pr_auc = average_precision_score(ytrue, ypred)

    plt.figure(figsize=(6, 3))
    plt.plot([], [], ' ', label=f'PR-AUC = {pr_auc:.2f}')
    plt.plot(thresholds, recall, color='red', label='Recall')
    plt.plot(thresholds, avg_recall, color='lightgray', label='Macro Average Recall')
    plt.plot(thresholds, f1, color='steelblue', label='F1-score')
    plt.plot(thresh_max, f1[i_max], '*', color='black')

    x_lim_diff = np.abs(plt.gca().get_xlim()[0] - plt.gca().get_xlim()[1])
    y_lim_diff = np.abs(plt.gca().get_ylim()[0] - plt.gca().get_ylim()[1])

    plt.text(thresh_max, f1[i_max] - 0.1*y_lim_diff, f'F1={f1[i_max]:.2f}', ha='center')
    #plt.plot(thresholds, precision, label='precision')
    plt.title(f'Score Plots {title}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(loc=legend_loc)
    if save_plot:
        title_save = clean_title(title)
        plt.savefig(f'Score Plots {title_save}.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    return thresh_max, pr_auc, recall[i_max], f1[i_max], avg_recall[i_max]


    #%%
def coef_abs_mean(coef, group_labels, original_labels):
    df_coef = pd.DataFrame(np.abs(coef[:,0]), index=group_labels)
    df_coef = df_coef.groupby(df_coef.index).mean().T
    df_coef.columns = original_labels
    print("=============================================\nMean of absolute values of group coefficients\n=============================================")
    print(df_coef)

    #%%
def plot_sigmoid_with_preds(ytrue, ypred, sample0=0.02, thresh=False, title=[], save_plot=False):
    import matplotlib.colors as mcolors
    colors = ["steelblue", "red"]
    cmap_name = "steelblue_red"
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

    yt = ytrue.reset_index(drop=True).copy()

    i_simoid1 = (yt == 1).values
    y_test1 = yt[i_simoid1]

    i_simoid0 = (yt == 0).values
    y_test0 = yt[i_simoid0].sample(frac=sample0, random_state=0)

    y_test_sample = pd.concat((y_test1, y_test0), axis=0).sample(frac=1, random_state=0)
    y_pred_sample = ypred[y_test_sample.index]

    y_preds = y_pred_sample
    y_test = y_test_sample.values

    # Generate a range of values for x to plot the sigmoid function
    x_values = np.linspace(0, 1, len(y_preds))

    # Compute the sigmoid of x_values
    sigmoid_values = expit(10 * (x_values - 0.5))

    plt.figure(figsize=(6, 3))

    # Plot y_preds with colors based on y_test
    plt.scatter(x_values, y_preds, c=y_test, cmap=custom_cmap, s=10, alpha=0.5)
    plt.scatter([], [], color='steelblue', s=15, marker='o', label=f'Class 0')
    plt.scatter([], [], color='red', s=15, marker='o', label=f'Class 1')

    # Plot optimal threshold
    if thresh:
        plt.hlines(thresh, 0, 1, linestyles='dashed', colors='black', linewidth=2, label='Threshold')

    # Plot the sigmoid function
    plt.plot(x_values, sigmoid_values, label='Sigmoid Function', color='black')
    plt.xlabel('Data Points')
    plt.ylabel('Predicted Probabilities')
    plt.title(f'Predictions {title}')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(loc='lower right')
    if save_plot:
        title_save = clean_title(title)
        plt.savefig(f'Sigmoid {title_save}.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

#%%
def full_class_report(ytrue, ypred, title=[], l_lim=0, u_lim=1, num_thresh=50, criterion='f1', legend_loc='best', sample0=0.03, save_plot=False):

    thresh_opt, _, _, _, _ = classification_plots(ytrue, ypred, l_lim=l_lim, u_lim=u_lim, num_thresh=num_thresh, criterion=criterion, title=title, legend_loc=legend_loc, save_plot=save_plot)

    plot_sigmoid_with_preds(ytrue, ypred, sample0=sample0, thresh=thresh_opt, title=title, save_plot=save_plot)

    # Generate predictions
    yp = [1 if i > thresh_opt else 0 for i in ypred]

    # Producce confusion matrix
    conf = confusion_matrix(ytrue, yp)
    conf_df = pd.DataFrame(conf)

    # produce classification report
    report = classification_report(ytrue, yp)

    def printed(output_str):
        print(output_str)

    def printed_and_saved(output_str, file):
        print(output_str)
        file.write(output_str + '\n')

    # clean title
    title_save = clean_title(title)

    # title
    title_str = f'#########{title_save[1:]} #########'

    # confustion matrix title
    conf_str = f'============================================\nConfusion Matrix - Optimal Threshold = {thresh_opt:.3f}\n============================================\n'

    # classification report title
    report_str = f'\n=================================================\nClassification Report - Optimal Threshold = {thresh_opt:.3f}\n=================================================\n'


    # Open a text file for writing
    if save_plot:
        with open(f'{title_save[1:]}.txt', 'w') as file:
            # title
            printed_and_saved(title_str, file)
            # conf matrix
            printed_and_saved(conf_str, file)
            printed_and_saved(conf_df.to_string(index=False), file)
            # Classification report
            printed_and_saved(report_str, file)
            printed_and_saved(report, file)
    else:
        # title
        printed(title_str)
        # conf matrix
        printed(conf_str)
        printed(conf_df.to_string(index=False))
        # Classification report
        printed(report_str)
        printed(report)


#%%
'''def assign_quantile_group(row, group, quantiles, column_name):
    col = row[column_name]
    value = group[col]

    # Initialize the group number (category) as the first category
    group_category = f"1"

    # Iterate over the quantiles and assign the group category
    for i, q in enumerate(quantiles):
        if value >= q:
            group_category = f"{i + 2}"  # Group names start from "Group 1"
        else:
            break  # Once the value is less than the quantile, stop checking

    return group_category'''

#%%

def best_recall_thresh(ytrue, ypred, key_word= '', fn_limit=10, l_lim=0, u_lim=1, num_thresh=1000):
    FP = []
    FN = []
    opt_thresh = []

    thresholds = np.linspace(l_lim, u_lim, num_thresh)

    for thresh in thresholds:
        y_thresh = [1 if i > thresh else 0 for i in ypred]
        _, fp, fn, _ = confusion_matrix(ytrue, y_thresh).ravel()

        if fn <= fn_limit:
            FP.append(fp)
            FN.append(fn)
            opt_thresh.append(thresh)
    i_max = np.argmax(FN)
    best_recall = {
        f'fp_{key_word}': FP[i_max],
        f'fn_{key_word}': FN[i_max],
        f'thresh_{key_word}': opt_thresh[i_max]
    }

    return best_recall


#%%

def model_fp_plots(recall_dict, top_models=2, save_plot=False, fig_size=(8,4)):

    for key, value in recall_dict.items():
        globals()[key] = value

    best_fp = np.inf
    colors = ['red', 'lightgray', 'steelblue']
    models = ['r', 'svm', 'xgb']
    datatyp = ['original', 'original_frac', 'original_c', 'nogroup', 'amount', 'coef', 'reduced', 'reduced_frac',
               'reduced_pb', 'balanced']

    labels_c = ['(1)\nOriginal\nVariabls',
                '(2)\nOriginal\nVariabls\nTrained on\n10%',
                '(3)\nOriginal\nVariabls -\nCust removed',
                '(4)\nCust, Merch,\nCat removed',
                '(5)\nCust,\nMerch, Cat\nGrpd Based\non Amount',
                '(6)\nCust,\nMerch, Cat\nGrpd Based\non Coefs',
                '(7)\nSame as (6)\nNew Features\nAdded.\nstep, age,\ngender removed',
                '(8)\nSame as (7)\nTrained on\n10%',
                '(9)\nSame as (7)\nTraining\nClass0 Slightly\nPulled Back',
                '(10)\nSame as (7)\nTraining\nBalanced']

    models_c = ['Ridge', 'SVM', 'XGBoost']
    all_fp = []
    all_ind = []
    fig1, ax = plt.subplots(figsize=fig_size)

    for i, model in enumerate(models):

        fp = [np.log10(eval(f'fp_{model}_{dtyp}', globals())) for dtyp in datatyp]

        ax.scatter(datatyp, fp, label=models_c[i], c=colors[i], marker='o', s=70, alpha=0.8)

        all_fp.append(fp)
        all_ind.append(range(len(datatyp)))

    all_fp = [round(10**x) if not np.isnan(x) else np.nan for inner_list in all_fp for x in inner_list]
    all_fp = np.array(all_fp).ravel()
    all_ind = np.array(all_ind).ravel()

    all_fp = dict(zip(all_fp, all_ind))
    all_fp_sort = sorted(all_fp.items(), key= lambda x: x[0])
    fp_top = all_fp_sort[:top_models]

    y_lim_diff = np.abs(plt.gca().get_ylim()[0] - plt.gca().get_ylim()[1])

    for top_no, dt in fp_top:
        ax.text(dt, np.log10(top_no)+((0.035*y_lim_diff)), f'FP={int(top_no)}', color='black', ha='center', rotation=0)


    # Formatter to display y-axis values with two decimal places
    formatter = FuncFormatter(lambda x, _: f'{int(10**x):,d}')

    # Apply the formatter to each axis
    ax.yaxis.set_major_formatter(formatter)

    # Set x-axis labels and titles for each figure
    ax.set_xticks(range(len(datatyp)))
    ax.set_xticklabels(labels_c)
    ax.set_ylabel('Class 0 Misclassifications', fontsize=12)
    ax.set_title('Class 0 Misclassifications (FP) for Thresholds Allowing 10 or Less Class 1 Misclassifications (FN)')
    ax.grid(linestyle='--', alpha=0.5)
    ax.legend()
    if save_plot:
        fig1.savefig('best_recall_plots.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()


#%%

def model_score_plots(scores, save_plot=False, fig_size=(8,4)):

    for key, value in scores.items():
        globals()[key] = value

    colors = ['red', 'lightgray', 'steelblue'] #['pink', 'red', 'lightgray', 'steelblue']
    models = ['r', 'svm', 'xgb'] #['glass', 'r', 'svm', 'xgb']
    datatyp = ['original', 'original_frac', 'original_c', 'nogroup', 'amount', 'coef', 'reduced', 'reduced_frac',
               'reduced_pb', 'balanced']

    labels_c = ['(1)\nOriginal\nVariabls',
                '(2)\nOriginal\nVariabls\nTrained on\n10%',
                '(3)\nOriginal\nVariabls -\nCust removed',
                '(4)\nCust, Merch,\nCat removed',
                '(5)\nCust,\nMerch, Cat\nGrpd Based\non Amount',
                '(6)\nCust,\nMerch, Cat\nGrpd Based\non Coefs',
                '(7)\nSame as (6)\nNew Features\nAdded.\nstep, age,\ngender removed',
                '(8)\nSame as (7)\nTrained on\n10%',
                '(9)\nSame as (7)\nTraining\nClass0 Slightly\nPulled Back',
                '(10)\nSame as (7)\nTraining\nBalanced']

    models_c = ['Ridge', 'SVM', 'XGBoost'] #['G-LASSO', 'Ridge', 'SVM', 'XGBoost']

    fig1, ax = plt.subplots(figsize=fig_size)
    fig2, bx = plt.subplots(figsize=fig_size)
    fig3, cx = plt.subplots(figsize=fig_size)
    fig4, dx = plt.subplots(figsize=fig_size)

    for i, model in enumerate(models):
        prauc = [eval(f'pr_auc_{model}_{dtyp}', globals()) for dtyp in datatyp]
        rc = [eval(f'recall_{model}_{dtyp}', globals()) for dtyp in datatyp]
        f1 = [eval(f'f1_{model}_{dtyp}', globals()) for dtyp in datatyp]
        recall = [eval(f'recall_macro_{model}_{dtyp}', globals()) for dtyp in datatyp]

        ax.plot(datatyp, prauc, label=models_c[i], c=colors[i], marker='o', markersize=5)
        bx.plot(datatyp, rc, label=models_c[i], c=colors[i], marker='o', markersize=5)
        cx.plot(datatyp, f1, label=models_c[i], c=colors[i], marker='o', markersize=5)
        dx.plot(datatyp, recall, label=models_c[i], c=colors[i], marker='o', markersize=5)

    # Formatter to display y-axis values with two decimal places
    formatter = FuncFormatter(lambda x, _: f'{x:.2f}')

    # Apply the formatter to each axis
    ax.yaxis.set_major_formatter(formatter)
    bx.yaxis.set_major_formatter(formatter)
    cx.yaxis.set_major_formatter(formatter)
    dx.yaxis.set_major_formatter(formatter)

    # Set the y-axis limits for all plots
    ax.set_ylim(0.63, 1.02)
    bx.set_ylim(0.63, 1.02)
    cx.set_ylim(0.63, 1.02)
    dx.set_ylim(0.63, 1.02)

    # Set x-axis labels and titles for each figure
    ax.set_xticks(range(len(datatyp)))
    ax.set_xticklabels(labels_c)
    ax.set_title('PR-AUC')
    ax.grid(linestyle='--', alpha=0.5)
    ax.legend()
    if save_plot:
        fig1.savefig('comp_prauc_score_plot.png', bbox_inches='tight', pad_inches=0.1)

    bx.set_xticks(range(len(datatyp)))
    bx.set_xticklabels(labels_c)
    bx.set_title('Recall')
    bx.grid(linestyle='--', alpha=0.5)
    bx.legend()
    if save_plot:
        fig2.savefig('comp_recall_score_plot.png', bbox_inches='tight', pad_inches=0.1)

    cx.set_xticks(range(len(datatyp)))
    cx.set_xticklabels(labels_c)
    cx.set_title('F1 Score')
    cx.grid(linestyle='--', alpha=0.5)
    cx.legend()
    if save_plot:
        fig3.savefig('comp_f1_score_plot.png', bbox_inches='tight', pad_inches=0.1)

    dx.set_xticks(range(len(datatyp)))
    dx.set_xticklabels(labels_c)
    dx.set_title('Recall Macro')
    dx.grid(linestyle='--', alpha=0.5)
    dx.legend()
    if save_plot:
        fig4.savefig('comp_recall_macro_score_plot.png', bbox_inches='tight', pad_inches=0.1)

    # Show each plot after saving
    plt.show()


#%%