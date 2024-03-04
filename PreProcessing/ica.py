import os
import mne
mne.set_log_level('error')
from autoreject import get_rejection_threshold

def ica_prepare_data(raw):
    # Filter settings
    ica_low_cut = 1.0       # ICA is sensitive to low-frequency drifts and therefore requires the data to be high-pass filtered prior to fitting. Typically, a cutoff frequency of 1 Hz is recommended.
    ica_hi_cut  = 30

    raw_ica = raw.copy().filter(ica_low_cut, ica_hi_cut)

    return raw_ica

def independent_comp_analysis(raw_ica):
    # Break raw data into 1s epochs
    tstep = 1.0
    events_ica = mne.make_fixed_length_events(raw_ica, duration=tstep)
    epochs_ica = mne.Epochs(raw_ica, events_ica,
                            tmin=0.0, tmax=tstep,
                            baseline=None,
                            preload=True)

    reject = get_rejection_threshold(epochs_ica);

    # ICA parameters
    random_state = 42   # ensures ICA is reproducable each time it's run
    ica_n_components = 0.99    # Specify n_components as a decimal to set % explained variance

    # Fit ICA
    ica = mne.preprocessing.ICA(n_components=ica_n_components,
                                random_state=random_state,
                                )
    ica.fit(epochs_ica,
            reject=reject,
            tstep=tstep)
    
    ica_z_thresh = 1.96 
    eog_indices, eog_scores = ica.find_bads_eog(raw_ica, 
                                                ch_name=['Fp2'], 
                                                threshold=ica_z_thresh)
    ica.exclude = eog_indices

    return ica, epochs_ica

def ica_save(ica, p_id):
    if not os.path.exists(f"./ICA/{p_id}"):
        os.makedirs(f"./ICA/{p_id}")

        data_path = f"./ICA/"

        ica.save(data_path + p_id + '-ica.fif', 
                overwrite=True);




