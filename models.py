from used_packages import *


#%%

def glasso_paths(X_train, y_train, X_test, original_labels, c_start=-6, c_stop=2, c_num=10, l1_reg=0.05,
                 scoring='accuracy', no_groups=False, n_iter=100, tol=1e-5, cmap='Set1', title=[],
                 verbose=False, save_plot=False):
    from utils import clean_title, glasso_groups_func
    scaler_tr = StandardScaler()
    X_train_scaled = scaler_tr.fit_transform(X_train)
    scaler_ts = StandardScaler()
    X_test_scaled = scaler_ts.fit_transform(X_test)
    screr = get_scorer(scoring)
    count = 0
    group_lass_labels, labels = glasso_groups_func(
        original_labels,
        X_train.columns.tolist())
    if no_groups:
        groups = range(1, len(X_train.columns) + 1)
        group_plot = group_lass_labels
    else:
        groups = group_lass_labels
        group_plot = groups.copy()
    if verbose:
        print(f'===============\nOriginal Labels\n===============\n{labels}\n')
        print(f'===============\nGroup Numbers\n===============\n{groups}\n')
    coefs = []
    scores = []
    lambdas = np.logspace(c_start, c_stop, c_num)
    best_lambda = None
    best_score = float(0)
    np.random.seed(0)
    X_tr, X_ts, y_tr, y_ts = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=0)

    for alpha in lambdas:
        count += 1
        group_lasso = LogisticGroupLasso(groups=groups, group_reg=alpha, l1_reg=l1_reg,
                                         n_iter=n_iter,
                                         tol=tol,
                                         scale_reg="inverse_group_size",
                                         subsampling_scheme=1, warm_start=True)
        group_lasso.fit(X_tr, y_tr)
        coefs.append(group_lasso.coef_)
        scr = screr(group_lasso, X_ts, y_ts)
        if verbose:
            print(f'{count}/{c_num} lambda: {alpha:.2e} - {scoring} = {scr*100:.2f}%')
        scores.append(scr)
        if scr > best_score:
            best_score = scr
            best_lambda = alpha
            best_coefs = group_lasso.coef_
            best_glass_model = group_lasso
    best_ypred = best_glass_model.predict_proba(X_test_scaled)[:,1]

    if best_lambda is None:
        raise ValueError("Best lambda not found.")

    coefs = np.array(coefs)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap((cmap), len(labels))
    for i in range(coefs.shape[1]):
        color_index = group_plot[i]
        ax.plot(np.log(lambdas), coefs[:, i, 0], color=colors(color_index-1))
    ax.axvline(np.log(best_lambda), linestyle='--', color='black', label='Optimal lambda')

    lim = plt.gca().get_ylim()[0]
    lim_diff = np.abs(plt.gca().get_ylim()[0] - plt.gca().get_ylim()[1])
    x_lim_diff = np.abs(plt.gca().get_xlim()[0] - plt.gca().get_xlim()[1])

    ax.text(np.log(best_lambda)+0.01*x_lim_diff, (lim+(0.2*lim_diff)), f'Opt. Lambda = {best_lambda:.2e}', color='black', ha='left', rotation=0)
    ax.text(np.log(best_lambda)+0.01*x_lim_diff, (lim+(0.15*lim_diff)), f'Opt. {scoring} score = {best_score*100:.2f}%', color='red', ha='left', rotation=0)

    plt.xlabel('Log Lambda (**Optimal Lambda displayed, not Optimal log-Lambda**)')
    plt.ylabel('Scaled Coefficients')

    # Create proxy artists for the legend
    proxy_artists = [plt.Line2D([0], [0], color=colors(i), lw=2) for i in range(len(labels))]

    ax.set_title(f'Paths {title}')
    ax.legend(proxy_artists, labels, loc='upper right')
    if save_plot:
        title_save = clean_title(title)
        plt.savefig(f'G-LASS paths {title_save}.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    return best_glass_model, best_coefs, best_lambda, best_ypred, groups, labels, scores


#%%

def svm_model(X_train, y_train, X_test, y_test, C, kernel='poly', degree=3, bias=None, scoring='accuracy', class_weight=None, verbose=False):
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
    scaler_train = StandardScaler()
    X_scaled_train = scaler_train.fit_transform(X_train)
    scaler_test = StandardScaler()
    X_scaled_test = scaler_test.fit_transform(X_test)

    C = C
    bias = bias
    score_max = []
    conf_max = []
    bias_max = []
    acc_class_max = []
    C_plot = []
    bias_plot = []
    score_plot = []
    acc_class_plot = []
    for i, c in enumerate(C):
        score = []
        conf_matrix_all = []
        acc_class = []
        if verbose:
            print(f'{i+1}. C = {c}')
        for bia in bias:
            if verbose:
                print(f'    bias: {bia}')
            model_svm = SVC(C=c, kernel=kernel, degree=degree, coef0=bia, cache_size=1000, class_weight=class_weight, verbose=verbose).fit(X_scaled_train, y_train)
            y_p = model_svm.predict(X_scaled_test)
            scr = get_scorer(scoring)
            scr = scr(model_svm, X_scaled_test, y_test)
            co = confusion_matrix(y_test, y_p)
            score.append(scr)
            conf_matrix_all.append(co)
            acc_c = np.diag(co) / np.sum(co, axis=1)*100
            acc_class.append(acc_c.sum()/2)
            C_plot.append(c)
            bias_plot.append(bia)
            score_plot.append(scr)
            acc_class_plot.append(acc_c.sum()/2)

        indmx = np.argmax(score)
        acc_class_max.append(acc_class[indmx])
        score_max.append(score[indmx])
        conf_max.append(conf_matrix_all[indmx])
        bias_max.append(bias[indmx])

    ind_max = np.argmax(score_max)
    best_svm = SVC(C=C[ind_max], kernel=kernel, degree=degree, coef0=bias_max[ind_max], probability=True, cache_size=1000).fit(X_scaled_train, y_train)
    ypred_svm1 = best_svm.predict_proba(X_scaled_test)[:,1]
    dict_svm ={'score_opt': score_max[ind_max],
               'acc_class_opt': acc_class_max[ind_max],
               'conf_opt': conf_max[ind_max],
               'bias_opt': bias_max[ind_max],
               'C_opt': C[ind_max],
               'score_plot': score_plot,
               'acc_class_plot': acc_class_plot,
               'y_svm': ypred_svm1,
               }
    print(f'Optimal Values: {scoring} score: {score_max[ind_max]: .4f},  C: {C[ind_max]},   Bias: {bias_max[ind_max]}')
    return dict_svm

#%%
def xgb_logistic_model(X_train, y_train, X_test, y_test, eval_metric='auc', max_depth=3, threads=4, iter=10, scale_pos_weight=None, verbose=True):
    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.NaN)
    dtest = xgb.DMatrix(X_test, label=y_test, missing=np.NaN)

    param = {'max_depth': max_depth, 'eta': 1, 'objective': 'binary:logistic'}
    param['nthread'] = threads
    param['eval_metric'] = [eval_metric]

    if scale_pos_weight is not None:
        param['scale_pos_weight'] = scale_pos_weight

    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    num_round = iter
    bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=verbose)
    ypred = bst.predict(dtest)
    return bst, ypred


#%%
# this was adapted from group lasso model, some lasso references still persist but it doest mean l1 reg or group l1 reg is used
def ridge_paths(X_train, y_train, X_test, original_labels, c_start=0.1, c_stop=10, c_num=10,
                scoring='accuracy',
                solver="lbfgs", n_iter=100, tol=1e-5,
                cmap='Set1',
                title=[],
                verbose=False, save_plot=False):
    from sklearn.linear_model import LogisticRegression
    from utils import clean_title, glasso_groups_func
    scaler_tr = StandardScaler()
    X_train_scaled = scaler_tr.fit_transform(X_train)
    scaler_ts = StandardScaler()
    X_test_scaled = scaler_ts.fit_transform(X_test)
    count = 0
    group_plot, labels = glasso_groups_func(
        original_labels,
        X_train.columns.tolist())

    if verbose:
        print(f'===============\nOriginal Labels\n===============\n{labels}\n')
        print(f'===============\nGroup Numbers (Groups used for plotting purposes not for regularization)\n===============\n{group_plot}\n')

    coefs = []
    scores = []
    lambdas = np.logspace(c_start, c_stop, c_num)
    best_lambda = None
    best_score = float(0)
    np.random.seed(0)
    X_tr, X_ts, y_tr, y_ts = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=0)

    for alpha in lambdas:
        count += 1

        ridge_model = LogisticRegression(C=alpha, penalty='l2',
                                         random_state= 0,
                                         solver=solver,
                                         tol=tol,
                                         max_iter=n_iter)
        ridge_model.fit(X_tr, y_tr)
        coefs.append(ridge_model.coef_[0])
        scr = get_scorer(scoring)
        scr = scr(ridge_model, X_ts, y_ts)
        if verbose:
            print(f'{count}/{c_num} lambda: {1/alpha:.2e} - {scoring} = {scr*100:.2f}%')
        scores.append(scr)
        if scr > best_score:
            best_score = scr
            best_lambda = 1/alpha
            best_coefs = ridge_model.coef_
            best_ridge_model = ridge_model
    best_ypred = best_ridge_model.predict_proba(X_test_scaled)[:,1]

    if best_lambda is None:
        raise ValueError("Best lambda not found.")

    coefs = np.array(coefs)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap((cmap), len(labels))
    for i in range(coefs.shape[1]):
        color_index = group_plot[i]
        ax.plot(np.log(1/lambdas), coefs[:, i], color=colors(color_index-1))
    ax.axvline(np.log(best_lambda), linestyle='--', color='black', label='Optimal lambda')

    lim = plt.gca().get_ylim()[0]
    lim_diff = np.abs(plt.gca().get_ylim()[0] - plt.gca().get_ylim()[1])
    x_lim_diff = np.abs(plt.gca().get_xlim()[0] - plt.gca().get_xlim()[1])

    ax.text(np.log(best_lambda)+0.01*x_lim_diff, (lim+(0.2*lim_diff)), f'Opt. Lambda = {best_lambda:.2e}', color='black', ha='left', rotation=0)
    ax.text(np.log(best_lambda)+0.01*x_lim_diff, (lim+(0.15*lim_diff)), f'Opt. {scoring} score = {best_score*100:.2f}%', color='red', ha='left', rotation=0)

    plt.xlabel('Log Lambda')
    plt.ylabel('Scaled Coefficients')

    # Create proxy artists for the legend
    proxy_artists = [plt.Line2D([0], [0], color=colors(i), lw=2) for i in range(len(labels))]

    ax.set_title(f'Paths {title}')
    ax.legend(proxy_artists, labels, loc='upper right')
    if save_plot:
        title_save = clean_title(title)
        plt.savefig(f'Paths {title_save}.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    return best_ridge_model, best_coefs, best_lambda, best_ypred, group_plot, labels, scores


#%%
