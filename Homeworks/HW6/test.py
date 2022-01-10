import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from sklearn.linear_model import LogisticRegression
from numpy.testing import assert_equal


def add_difference(asimov_dataset):
    asimov_dataset['total_naughty_robots_previous_day'] =
    asimov_dataset['total_naughty_robots'].shift(1)

    asimov_dataset['change_in_naughty_robots'] =
    abs(asimov_dataset['total_naughty_robots_previous_day'] -
        asimov_dataset['total_naughty_robots'])

    return asimov_dataset[['total_naughty_robots', 'change_in_naughty_robots',
                           'robot_takeover_type']]


def test_change():
    asimov_dataset_input = pd.DataFrame({
        'total_naughty_robots': [1, 4, 5, 3],
        'robot_takeover_type': ['A', 'B', np.nan, 'A']
    })

    expected = pd.DataFrame({
        'total_naughty_robots': [1, 4, 5, 3],
        'change_in_naughty_robots': [np.nan, 3, 1, 2],
        'robot_takeover_type': ['A', 'B', np.nan, 'A']
    })

    result = add_difference(asimov_dataset_input)

    assert_frame_equal(expected, result)


def remove_nan_size(asimov_dataset):
    return asimov_dataset.dropna(subset=['robot_takeover_type'])


def clean_data(asimov_dataset):
    asimov_dataset_with_difference = add_difference(asimov_dataset)
    asimov_dataset_without_na = remove_nan_size(asimov_dataset_with_difference)

    return asimov_dataset_without_na


def test_cleanup():
    asimov_dataset_input = pd.DataFrame({
        'total_naughty_robots': [1, 4, 5, 3],
        'robot_takeover_type': ['A', 'B', np.nan, 'A']
    })

    expected = pd.DataFrame({
        'total_naughty_robots': [1, 4, 3],
        'change_in_naughty_robots': [np.nan, 3, 2],
        'robot_takeover_type': ['A', 'B', 'A']
    }).reset_index(drop=True)

    result = clean_data(asimov_dataset_input).reset_index(drop=True)

    assert_frame_equal(expected, result)

def get_reression_training_score(asimov_dataset, seed=9787):
    clean_set = clean_data(asimov_dataset).dropna()

    input_features = clean_set[['total_naughty_robots',
                                'change_in_naughty_robots']]
    labels = clean_set['robot_takeover_type']

    model = LogisticRegression(random_state=seed).fit(input_features, labels)
    return model.score(input_features, labels) * 100


def test_regression_score():
    asimov_dataset_input = pd.DataFrame({
        'total_naughty_robots': [1, 4, 5, 3, 6, 5],
        'robot_takeover_type': ['A', 'B', np.nan, 'A', 'D', 'D']
    })

    result = get_reression_training_score(asimov_dataset_input, seed=1234)
    expected = 40.0

    assert_equal(result, 50.0)
