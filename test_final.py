"""
Claire Li, CSE 163 AA
Yuhao Zhuang, CSE 163 AC
Runs MSE with actual output against expected output to see if
there is any difference
"""
import os

import final_163
import pandas as pd
import numpy as np
import matplotlib.image as mpimg

PLOTS = [
    "continent_medals.png",
    "events_change.png",
    "su_athletes_age_sex.png",
    "top_5_summer_golds.png",
    "top_10_teams_golds.png",
    "us_medals_by_sports.png",
]
EXPECTED_FUNCTIONS1 = [
    "top_10_teams_golds",
    "top_5_summer_gold",
    "events",
    "su_athletes_age_sex",
]
EXPECTED_FUNCTIONS2 = [
    "us_medals_by_sports",
]
EXPECTED_FUNCTIONS3 = [
    "continent_medals"
]
TEST_FILE = "test.csv"
OLYMPICS = 'olympics.csv'


def run_imgd(expected, actual):
    """
    Runs imgd of student output against expected.
    Produces diff image only if both student and expected output exist.
    """
    if not os.path.exists(actual):
        print(f"Could not find the file: {actual} after running \
              final_163.py\n")
    elif not os.path.exists(expected):
        print(f"Could not find the file: {expected}\n")
    else:
        print(f"Running image comparison tool on {actual}...")
        expected = mpimg.imread(expected)
        actual = mpimg.imread(actual)
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((expected.astype("float") - actual.astype("float")) ** 2)
        err /= float(expected.shape[0] * actual.shape[1])
        # print MSE
        if err == 0:
            print('There is no difference')
        else:
            print('There is difference', err)


def test_best_athlete(test_df, olympics):
    """
    Test the correctness of Q4, the athlete who got most medals.
    """
    result = final_163.most_medal_athlete(olympics).reset_index(drop=True)
    if not test_df.equals(result):
        print('not equal')


def main():
    data1 = pd.read_csv('olympics.csv', na_values=['---'])
    data2 = pd.read_csv('sports_events.csv', na_values=['---'])
    data3 = pd.read_csv('continents.csv', na_values=['---'])
    print("Checking for all functions:")
    for f in EXPECTED_FUNCTIONS1:
        try:
            getattr(final_163, f)(data1)
            print("    Found", f)
        except AttributeError:
            print("    ERROR: Missing", f, "in final_163.py")
    print()
    for f in EXPECTED_FUNCTIONS2:
        try:
            getattr(final_163, f)(data1, data2)
            print("    Found", f)
        except AttributeError:
            print("    ERROR: Missing", f, "in final_163.py")
    print()
    for f in EXPECTED_FUNCTIONS3:
        try:
            getattr(final_163, f)(data1, data3)
            print("    Found", f)
        except AttributeError:
            print("    ERROR: Missing", f, "in final_163.py")
    print()
    for plot_name in PLOTS:
        run_imgd(f"expected/{plot_name}", plot_name)
    # Test Q4
    test_df = pd.read_csv(TEST_FILE)  # dataframe
    olympics = pd.read_csv(OLYMPICS)  # dataframe
    test_best_athlete(test_df, olympics)


if __name__ == "__main__":
    main()
