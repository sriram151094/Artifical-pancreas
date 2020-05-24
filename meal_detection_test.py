#Testing the model to classify new data as Meal vs Non-meal
import os
import pickle
import scipy
import pywt
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import MinMaxScaler
from scipy.integrate import simps
from tensorflow.keras.models import load_model

def read_csv_file(filepath):
    d = []
    with open(filepath) as csvfile:
        areader = csv.reader(csvfile)
        max_elems = 0
        for row in areader:
            if max_elems < len(row): max_elems = len(row)
        csvfile.seek(0)
        for i, row in enumerate(areader):
            # fix my csv by padding the rows
            d.append(row + ["" for x in range(max_elems-len(row))])

    df = pd.DataFrame(d)
    return df


def convert_to_int(df_list):
    for df in df_list:
        for i in range(len(df.iloc[0])):
            df[i] = pd.to_numeric(df[i],errors='coerce')
    return df_list


def droplastcolumn(df_list):
    preprocessed_df_list = []
    for df in df_list:
        if df.shape[1]>=31:
            df = df.drop(df.columns[[30]], axis = 1)
        preprocessed_df_list.append(df)
    return preprocessed_df_list


def reversedf(df):
    df = df.iloc[:, ::-1]
    return df


def interpolatedf(df_list):
    preprocessed_df_list = []
    for df in df_list:
        for i in range(len(df)):
            df.loc[[i]] = df.loc[[i]].interpolate(axis=1, limit=None, limit_direction='both')
        preprocessed_df_list.append(df)
    return preprocessed_df_list


def dfdropna(df_list):
    preprocessed_df_list = []
    for df in df_list:
        df = df.dropna()
        df = df.reset_index(drop=True)
        preprocessed_df_list.append(df)
    return preprocessed_df_list


def construct_feature_matrix(top_fft_features, top_dwt_features, output_entropy,
                             feature_COV_, lognorm_mean_list, lognorm_std_list, feature_auc):
    Feature_Matrix = np.hstack(
        (top_fft_features[0][:, 1:9], top_dwt_features[:, 1:7], output_entropy[:, None], feature_COV_[:, None],
         lognorm_mean_list[:, None], lognorm_std_list[:, None], feature_auc[:, None]))

    return Feature_Matrix


# Function to extract Fast Fourier Transform given a Dataframe.
def calculate_fft(df_list):
    if len(df_list) == 1:
        return [np.abs(np.fft.fft(df_list[0]))]
    else:
        top_fft_features = []
        for df in zip(df_list):
            if len(top_fft_features) == 0:
                top_fft_features = np.abs(np.fft.fft(np.flip(df)))
            else:
                top_fft_features = np.concatenate((top_fft_features, np.abs(np.fft.fft(np.flip(df)))), axis=1)
        return top_fft_features


# Function to extract Windowed entropy given a Dataframe.
def windowed_entropy(df_list):
    output_entropy = []
    y = []
    ordered_cgm = []
    for df in df_list:
        for j in range(len(df)):
            temp = []
            temp1 = []
            c = df.iloc[j]
            for m in range(len(c) - 1, -1, -1):
                temp.append(c[m])
            y_array = np.array(temp)
            ordered_cgm.append(y_array)

    for i in range(len(ordered_cgm)):
        entropy_arr = []
        for j in range(1, 30, 5):
            s = scipy.stats.entropy(np.asarray(ordered_cgm)[i, j:j + 5])
            entropy_arr.append(s)
        output_entropy.append(np.amin(np.asarray(entropy_arr)))

    output_entropy = np.asarray(output_entropy)
    return output_entropy


# DWT Feature Calculation
def calc_feature_dwt(df):
    cA, cB = pywt.dwt(df, 'haar')
    cA_threshold = pywt.threshold(cA, np.std(cA) / 2, mode='soft')
    cB_threshold = pywt.threshold(cB, np.std(cB) / 2, mode='soft')

    reconstructed_signal = pywt.idwt(cA_threshold, cB_threshold, 'haar')
    feature_dwt_top8 = cA[:, :-8]  # sorted in Ascending

    return feature_dwt_top8


def discrete_wavelet_transform(df_list):
    top_dwt_features = []
    if len(df_list) == 1:
        return calc_feature_dwt(df_list[0])
    else:
        for df in df_list:
            if len(top_dwt_features) == 0:
                top_dwt_features = calc_feature_dwt(df)
            else:
                top_dwt_features = np.concatenate((top_dwt_features, calc_feature_dwt(df)))


# Function to get Coefficient of Variation Feature Calculation
def coef_of_variation(df_list):
    feature_COV = []

    for df in df_list:
        for i in range(len(df)):
            feature_COV.append(np.mean(df[i]) / np.std(df[i]))

    feature_COV_ = np.asarray(feature_COV)

    feature_COV_WO_nan = feature_COV_[np.isnan(feature_COV_) == False]
    feature_COV_WO_nan.sort()
    mean_with_threshold = np.mean(feature_COV_WO_nan[0:len(feature_COV_WO_nan) - 1])
    mean_with_threshold

    feature_COV_[feature_COV_ > 200] = mean_with_threshold
    for x in range(len(feature_COV_)):
        if np.isnan(feature_COV_[x]):
            feature_COV_[x] = mean_with_threshold

    return feature_COV_


def calculate_log_norm_distribution(df_list):
    lognorm_mean_list = []
    lognorm_std_list = []

    for df in df_list:
        for i in range(len(df)):
            x = df[i]
            x[x == 0] = np.mean(x)
            mu = np.mean(x)
            sigma = np.std(x)

            x_exp = x
            mu_exp = np.exp(mu)
            sigma_exp = np.exp(sigma)

            fitting_params_lognormal = scipy.stats.lognorm.fit(x_exp, floc=0, scale=mu_exp)
            lognorm_dist_fitted = scipy.stats.lognorm(*fitting_params_lognormal)
            t = np.linspace(np.min(x_exp), np.max(x_exp), 100)

            lognorm_dist = scipy.stats.lognorm(s=sigma, loc=0, scale=np.exp(mu))
            lognorm_mean_list.append(lognorm_dist.mean())
            lognorm_std_list.append(lognorm_dist.std())

    lognorm_std_list = np.asarray(lognorm_std_list)
    lognorm_mean_list = np.asarray(lognorm_mean_list)
    return [lognorm_mean_list, lognorm_std_list]


def area_under_curve(df_list):
    feature_auc = []

    for df in df_list:
        for x in simps(df[:, ::-1], dx=5):
            feature_auc.append(x)

    feature_auc = np.asarray(feature_auc)
    return feature_auc


def main():
    print("Running MEAL DETECTION test script..\n")
    fpath = input("Enter the dataset absolute path :")
    if not os.path.isfile(fpath):
        print("File does not exists")
        return -1

    TestData = read_csv_file(fpath)

    # Pre-process Test data
    pre_processed_test_data = convert_to_int([TestData])
    pre_processed_test_data = droplastcolumn(pre_processed_test_data)
    pre_processed_test_data = interpolatedf(pre_processed_test_data)
    pre_processed_test_data = dfdropna(pre_processed_test_data)
    pre_processed_test_data = reversedf(pre_processed_test_data[0])

    entropy_test = windowed_entropy([pre_processed_test_data])

    scaler = MinMaxScaler(feature_range=(0, 1))
    pre_processed_test_data = scaler.fit_transform(pre_processed_test_data)

    # Calculate FFT
    fft_test = calculate_fft([pre_processed_test_data])
    dwt_test = discrete_wavelet_transform([pre_processed_test_data])
    coef_var_test = coef_of_variation([pre_processed_test_data])
    log_norm_res = calculate_log_norm_distribution([pre_processed_test_data])
    feature_auc = area_under_curve([pre_processed_test_data])


    # Get feature matrix of Test data
    test_f_matrix = construct_feature_matrix(fft_test, dwt_test, entropy_test, coef_var_test, log_norm_res[0], log_norm_res[1], feature_auc)
    scaled_test_f_matrix = scaler.fit_transform(test_f_matrix)
    scaled_test_f_matrix_ = np.nan_to_num(scaled_test_f_matrix)

    final_test_matrix = np.asmatrix(scaled_test_f_matrix_)
    # Apply PCA to test data matrix by loading the trained PCA model
    filename = './Models/pca_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    reduced_test_matrix = loaded_model.transform(final_test_matrix)

    filename = "./Models/knn_model.sav"
    knn_clf = pickle.load(open(filename, 'rb'))

    filename = "./Models/rand_for_model.sav"
    rand_for_clf = pickle.load(open(filename, 'rb'))

    filename = "./Models/clf1_model.sav"
    ada_boost_clf = pickle.load(open(filename, 'rb'))

    filename = "./Models/clf_model.sav"
    grad_boost_clf = pickle.load(open(filename, 'rb'))

    filename = "./Models/svm_model.sav"
    svm_clf = pickle.load(open(filename, 'rb'))

    filename = "./Models/gnb_model.sav"
    gnb_clf = pickle.load(open(filename, 'rb'))

    filename = "./Models/d_tree_model.sav"
    d_tree_clf = pickle.load(open(filename, 'rb'))

    ann_model = load_model('./Models/ann_model.h5')

    y_pred_results = {}
    y_pred = knn_clf.predict(reduced_test_matrix)
    y_pred_results["knn"] = y_pred

    y_pred = rand_for_clf.predict(reduced_test_matrix)
    y_pred_results["random_forest"] = y_pred

    y_pred = ada_boost_clf.predict(reduced_test_matrix)
    y_pred_results["ada_boost"] = y_pred

    y_pred = grad_boost_clf.predict(reduced_test_matrix)
    y_pred_results["grad_boost"] = y_pred

    y_pred = svm_clf.predict(reduced_test_matrix)
    y_pred_results["svm"] = y_pred

    y_pred = gnb_clf.predict(reduced_test_matrix)
    y_pred_results["gnb"] = y_pred

    y_pred = d_tree_clf.predict(reduced_test_matrix)
    y_pred_results["decision_tree"] = y_pred

    # y_pred = ann_model.predict(reduced_test_matrix)
    y_preds = ann_model.predict(reduced_test_matrix)
    y_list = []
    for i in y_preds:
        if i > 0.5:
            y_list.append(1)
        else:
            y_list.append(0)
            
    y_pred_results["Neural_Network"] = y_pred

    df_result = pd.DataFrame().from_dict(y_pred_results)

    print(df_result)

    df_result.to_csv("predictions.csv", encoding='utf-8')


if __name__ == '__main__':
    main()