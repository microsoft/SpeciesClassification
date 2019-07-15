import pandas as pd

FILE1 = 'inc4_488_test_result.csv'
FILE2 = 'resnext101_488_test_result.csv'
SAVE_TO = 'submission.csv'


def main(file1, file2):
    '''Average out the results of the model.'''
    # Read both the files and index the filename
    df1 = pd.read_csv(file1).set_index('filename')
    df2 = pd.read_csv(file2).set_index('filename')

    # Average the score of two files
    df = (df1 + df2) / 2
    df = df.reset_index()
    df.to_csv(SAVE_TO, index=False)


if __name__ == '__main__':
    main(FILE1, FILE2)
