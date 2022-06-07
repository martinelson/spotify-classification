import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

datasets = ['track_features_shuffled.csv', 'track_features_2_shuffled.csv', 'track_features_3_shuffled.csv',
            'track_features_4_shuffled.csv']
plot_paths = ['plots/', 'plots_2/', 'plots_3/', 'plots_4/']
feature_cols = ['Hip Hop', 'R&B', 'Dance', 'Country', 'Blues', 'Reggae', 'Folk', 'Pop', 'Rock', 'Jazz']
feature_cols_combo = ['Hip Hop/Pop', 'R&B/Reggae', 'Rock/Dance', 'Blues/Jazz', 'Folk/Country']

for data_path, plot_path in zip(datasets, plot_paths):
    # Loading in data and splitting into feature and label vectors
    data = pd.read_csv(data_path)
    cols = list(data.keys())
    cols = cols[:-1]
    y = data.iloc[:, -1:].to_numpy().ravel()
    X = data.iloc[:, :-1].to_numpy()
    genres = feature_cols
    if data_path == 'track_features_3_shuffled.csv':
        genres = feature_cols_combo
    # Scaling data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=88)

    # Naming and creating classifiers
    name = ['KNN', 'Log Reg', 'Neural Network', 'SVM', 'kernel SVM']
    classifiers = [
        KNeighborsClassifier(n_neighbors=4),
        LogisticRegression(max_iter=1000, random_state=88),
        MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=2000, random_state=8),
        SVC(random_state=8),
        SVC(kernel='rbf', degree=5, random_state=8),
    ]

    # Implementing initial CART to find max depth
    cart_clf = tree.DecisionTreeClassifier()
    cart_clf.fit(X_train, y_train)
    cart_misclass = 1 - cart_clf.score(X_test, y_test)
    max_depth = cart_clf.tree_.max_depth

    # Running random forest to find lowest misclass rate at certain n_estimator value
    misclass_rate = []
    for i in range(100):
        rf_clf = RandomForestClassifier(n_estimators=i+1, random_state=0, max_depth=max_depth)
        rf_clf.fit(X_train, y_train)
        accuracy = rf_clf.score(X_test, y_test)
        misclass_rate.append(1-accuracy)

    min_rate = min(misclass_rate)

    min_estimator = misclass_rate.index(min_rate) + 1

    # Appending Random forest classifier to list of classifiers
    classifiers.append(RandomForestClassifier(n_estimators=min_estimator, max_depth=max_depth))
    name.append('Random Forest')

    # Creating confusion matrix plot and displays accuracy from different models
    def model_comparison(clf_list, name_list, y_test, y_train, X_test, X_train, path, labels):
        for name, clf in zip(name_list, clf_list):
            report_labels = np.unique(y_test)
            y_pred = clf.fit(X_train, y_train).predict(X_test)
            c_matrix = confusion_matrix(y_test, y_pred, labels=report_labels)
            c_matrix_df = pd.DataFrame(c_matrix, index=[i for i in labels], columns=[i for i in labels])
            accuracy = round(clf.score(X_test, y_test), 2)
            plt.figure(figsize=(11, 9))
            plt.title(f'{name} - Accuracy: {accuracy}')
            sns.heatmap(c_matrix_df, annot=True)
            file_path = path+f'/{name}_cm.png'
            plt.savefig(file_path)
            plt.close()

    # Calling model compare function for initial models
    # model_comparison(classifiers, name, y_test, y_train, X_test, X_train, f'{plot_path}initial_cm', genres)

    # Cross validating different models based on various parameters
    def cross_validation(model, params, cv_int, x, y):
        cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=cv_int)
        cv.fit(x, y)
        best_model = cv.best_estimator_
        return best_model


    cross_val_num = 5
    # KNN CV
    n_neighbors_list = list(range(1, 10))
    knn_model = cross_validation(KNeighborsClassifier(), {'n_neighbors': n_neighbors_list},
                                 cross_val_num, X_train, y_train)

    # SVC Kernel CV
    degree_list = list(range(3, 30))
    svc_kernel_model = cross_validation(SVC(kernel='rbf', random_state=8),
                                        {'degree': degree_list}, cross_val_num, X_train, y_train)

    # SVC CV - Creating C parameter list
    def decimal_squence():
        i = 1.0
        s = []
        while i < 3.2:
            s.append(i)
            i += 0.2
            i = round(i, 1)
        return s


    C_list = decimal_squence()
    svc_model = cross_validation(SVC(random_state=8), {'C': C_list}, cross_val_num, X_train, y_train)

    # Log Reg CV
    solver_list = ['newton-cg', 'lbfgs', 'liblinear']
    log_model = cross_validation(LogisticRegression(max_iter=1000, random_state=88),
                                 {'C': C_list, 'solver': solver_list}, cross_val_num, X_train, y_train)

    # MLP CV
    hls_list = [(100,), (10, 5), (20, 10), (30, 20), (40, 30)]
    nn_model = cross_validation(MLPClassifier(max_iter=2000, activation='identity', random_state=8),
                                {'hidden_layer_sizes': hls_list}, cross_val_num, X_train, y_train)

    # RF CV
    n_list = list(range(1, 100))
    rf_model = cross_validation(RandomForestClassifier(random_state=0), {'n_estimators': n_list},
                                cross_val_num, X_train, y_train)

    # Creating cross validated model list for future comparison
    cv_classifiers = [knn_model, log_model, nn_model, svc_model, svc_kernel_model, rf_model]

    # PCA plots per model
    max_components = X.shape[1]
    cv_classifiers.pop(5)

    for n, clf in zip(name, cv_classifiers):
        test_accuracy = []
        for num in range(max_components):
            pca = PCA(n_components=num+1)
            X_train_split = pca.fit_transform(X_train)
            X_test_split = pca.transform(X_test)
            clf.fit(X_train_split, y_train)
            accuracy = round(clf.score(X_test_split, y_test), 2)
            test_accuracy.append(accuracy)
        plt.figure(figsize=(11, 9))
        plt.title(f'{n} PCA')
        plt.xlabel('# of Components')
        plt.ylabel('accuracy')
        component_range = list(range(1, max_components+1))
        plt.plot(component_range, test_accuracy)
        # plt.savefig(f'{plot_path}pca/{n}_pca.png')
        plt.close()

    # Feature importance list from Random Forest
    rf_model.fit(X_train, y_train)
    feature_order = rf_model.feature_importances_
    df_importances = pd.DataFrame({'Importance': feature_order, 'Variable': cols}).sort_values('Importance',
                                                                                               ascending=False)
    df_importances['Cusum'] = df_importances['Importance'].cumsum(axis=0)

    if data_path == 'track_features_shuffled.csv':
        x = df_importances['Variable'].to_numpy()
        y = df_importances['Importance'].to_numpy()
        plt.figure(figsize=(11, 9))
        plt.title("Random Forest Feature Importance")
        sns.barplot(x=x, y=y, palette="crest")
        plt.xticks(rotation=45)
        plt.savefig(f'{plot_path}feature_imp.png')
        plt.close()

# gb_model = GradientBoostingClassifier(random_state=8)
# params = {'n_estimators': [100, 200], 'subsample': [.9, 1], 'max_depth': [2, 4],
#           'min_samples_split': [2, 3], 'min_samples_leaf': [1, 2], 'max_features': [4, 8, 10]}
# gb_model = cross_validation(gb_model, params, cross_val_num, X_train, y_train)
    max_features = X.shape[1]
    gb_model = GradientBoostingClassifier(max_depth=10, max_features=max_features,
                                          min_samples_split=4, subsample=1, random_state=8)
    cv_classifiers.append(gb_model)
    cv_classifiers.append(rf_model)
    name.append('GB')
    # model_comparison(cv_classifiers, name, y_test, y_train, X_test, X_train, f'{plot_path}cv_cm', genres)
