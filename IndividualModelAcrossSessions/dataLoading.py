import mne
import numpy as np

def filepath(subject, session, run, annotated):
    if annotated == True:
    #    data_dir = f"/Users/sunniva/Documents/Alcohol_Detection_Project_2023-2024/PreProcessing/Preprocessed/"
    #    f_format = "-processed.fif"
        data_dir = f"../Subjects/"
        f_format = '_eeg' + '-annotated-raw.fif'
    else:
        data_dir = f"../Subjects/"
        f_format = '_eeg' + '.fif'
    f_name = f"sub-{subject}/ses-S00{session+1}/" + f'sub-{subject}_ses-S00' + str(session+1) + '_task-Default_run-00' + str(run) # Data recorded using unicorn system
    return data_dir + f_name + f_format

def epoch_and_label_file(filepath, epoch_duration):
    """
    Reads a .fif file, segments it into 5-second epochs, and assigns a label.
    
    Parameters:
    - filepath: Path to the .fif file.
    - label: 0 for placebo, 1 for alcohol.
    
    Returns:
    - epochs_data: Array of epoch data.
    - labels: Array of labels corresponding to each epoch.
    """

    raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=False)
    # print(epochs.get_data())
    print(epochs[0].get_data(), len(epochs[0].get_data()))
    epochs_data = epochs.get_data()

    return epochs_data

def add_to_dataset(dataset, key, data):
    """
    Add processed epochs to the dataset under the given key.
    """
    if key in dataset:
        dataset[key].append(data)
    else:
        dataset[key] = [data]