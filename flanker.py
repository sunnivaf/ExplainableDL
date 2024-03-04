import expyriment
import pylsl

n_trials_block = 16
n_blocks = 6
durations = 2000
flanker_stimuli =  ["XXXXX", "XXCXX", "CCXCC", "CCCCC", "VVVVV", "VVBVV", "BBVBB", "BBBBB", "XXVXX", "XXBXX", "CCVCC", "CCBCC", "VVXVV", "VVCVV", "BBXBB", "BBCBB"]
instructions = "You will see 5 letters at a time. You need to respond to the one in the middle. \n If you see an X or a C, press the A key. If you see a V or a B, press the L key. \n For example, if you see the letters XXCXX, you press the A key on your keyboard. \n Press the space bar to start the test."

#LSL stream
info = pylsl.stream_info('Flanker','Markers', 1, 0, pylsl.cf_string, 'unsampledStream')
outlet = pylsl.stream_outlet(info)

experiment = expyriment.design.Experiment(name='Flanker Task')
expyriment.control.initialize(experiment)

for block in range(n_blocks):
    temp_block = expyriment.design.Block(name=str(block + 1))
    for trial in range(n_trials_block):
        curr_stim = flanker_stimuli[trial]
        temp_stim = expyriment.stimuli.TextLine(text=curr_stim, text_size=40)
        temp_trial = expyriment.design.Trial()
        temp_trial.add_stimulus(temp_stim)
        if trial <= 7: #First half of stimuli list
            trialtype = 'congurent'
        elif trial > 7: #Second half of stimuli list
            trialtype = 'incogurent'
        if curr_stim[2] == 'X' or curr_stim[2] == 'C':
            correctresponse = 97 #ASCII value A
        elif curr_stim[2] == 'V' or curr_stim[2] == 'B':
            correctresponse = 108 #ASCII value L
        temp_trial.set_factor('trialtype', trialtype)
        temp_trial.set_factor('correctresponse', correctresponse)
        temp_block.add_trial(temp_trial)
    temp_block.shuffle_trials()
    experiment.add_block(temp_block)

experiment.data_variable_names = ["block", "correctresp", "response", "trial", "RT", "accuracy", "trialtype"]
expyriment.control.start(skip_ready_screen=True)
fixation_cross = expyriment.stimuli.FixCross()

expyriment.stimuli.TextScreen('Flanker task', instructions).present()
experiment.keyboard.wait(expyriment.misc.constants.K_SPACE)
mrk = pylsl.vectorstr(['5']) #Mark start of Flanker by pressing space
outlet.push_sample(mrk)
experiment.clock.wait(1000) #Delay start for 1s to get start marker

for block in experiment.blocks:
    for trial in block.trials:
        mrk = pylsl.vectorstr(['1']) #Mark for cross

        outlet.push_sample(mrk)
        fixation_cross.present()
        trial.stimuli[0].preload()
        experiment.clock.wait(durations)
        if trial.get_factor('trialtype') == 'congurent':
            mrk = pylsl.vectorstr(['2']) #Mark for congurent
                
        else:
            mrk = pylsl.vectorstr(['3']) #Mark for incongurent

        outlet.push_sample(mrk)
        trial.stimuli[0].present()
        experiment.clock.reset_stopwatch()
        key, rt = experiment.keyboard.wait(keys=[expyriment.misc.constants.K_a, expyriment.misc.constants.K_l], duration=durations)
        mrk = pylsl.vectorstr([str(key)])
        outlet.push_sample(mrk)
        experiment.clock.wait(durations - experiment.clock.stopwatch_time)

        if key == trial.get_factor('correctresponse'):
            acc = 1
        else:
            acc = 0
            
        experiment.data.add([block.name, trial.get_factor('correctresponse'), key, trial.id, rt, acc, trial.get_factor('trialtype')])

    if block.name != '6':
        mrk = pylsl.vectorstr(['4']) #Mark for break
        outlet.push_sample(mrk)
        expyriment.stimuli.TextScreen("Short break", "That was block " + block.name + " of " + str(n_blocks) + ". \n Next block will start soon.").present()

        experiment.clock.wait(3000)
mrk = pylsl.vectorstr(['6']) #Mark for finished Flanker
outlet.push_sample(mrk)
expyriment.control.end(goodbye_text="You are now finished with the Flanker test. Great job!", goodbye_delay=10000)
print(experiment.data)
