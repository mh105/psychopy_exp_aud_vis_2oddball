#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.2a1),
    on Tue Mar 17 13:54:14 2026
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '4'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from eeg
import pyxid2
import threading
import signal


def exit_after(s):
    '''
    function decorator to raise KeyboardInterrupt exception
    if function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, signal.raise_signal, args=[signal.SIGINT])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


@exit_after(1)  # exit if function takes longer than 1 seconds
def _get_xid_devices():
    return pyxid2.get_xid_devices()


def get_xid_devices():
    print("Getting a list of all attached XID devices...")
    attempt_count = 0
    while attempt_count >= 0:
        attempt_count += 1
        print('     Attempt:', attempt_count)
        attempt_count *= -1  # try to exit the while loop
        try:
            devices = _get_xid_devices()
        except KeyboardInterrupt:
            attempt_count *= -1  # get back in the while loop
    return devices


devices = get_xid_devices()

if devices:
    dev = devices[0]
    print("Found device:", dev)
    assert dev.device_name in ['Cedrus C-POD', 'Cedrus StimTracker Quad'], "Incorrect XID device detected."
    if dev.device_name == 'Cedrus C-POD':
        pod_name = 'C-POD'
    else:
        pod_name = 'M-POD'
    dev.set_pulse_duration(50)  # set pulse duration to 50ms

    # Start EEG recording
    print("Sending trigger code 126 to start EEG recording...")
    dev.activate_line(bitmask=126)  # trigger 126 will start EEG
    print("Waiting 10 seconds for the EEG recording to start...\n")
    core.wait(10)  # wait 10s for the EEG system to start recording

    # Marching lights test
    print(f"{pod_name}<->eego 7-bit trigger lines test...")
    for line in range(1, 8):  # raise lines 1-7 one at a time
        print("  raising line {} (bitmask {})".format(line, 2 ** (line-1)))
        dev.activate_line(lines=line)
        core.wait(0.5)  # wait 500ms between two consecutive triggers
    dev.con.set_digio_lines_to_mask(0)  # XidDevice.clear_all_lines()
    print("EEG system is now ready for the experiment to start.\n")

else:
    # Dummy XidDevice for code components to run without C-POD connected
    class dummyXidDevice(object):
        def __init__(self):
            pass
        def activate_line(self, lines=None, bitmask=None):
            pass


    print("WARNING: No C/M-POD connected for this session! "
          "You must start/stop EEG recording manually!\n")
    dev = dummyXidDevice()

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.2a1'
expName = 'visual_oddball_2ob'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s/%s_%s_%s' % (expInfo['participant'], expInfo['participant'], expName, expInfo['session'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/alexhe/Library/CloudStorage/Dropbox/Active_projects/PsychoPy/exp_aud_vis_2oddball/visual_oddball_2ob.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # log the filename of last_app_load.log
    print('target_last_app_load_log_file: ' + filename + '_last_app_load.log')
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('debug')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_welcome') is None:
        # initialise key_welcome
        key_welcome = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_welcome',
        )
    # create speaker 'read_welcome'
    deviceManager.addDevice(
        deviceName='read_welcome',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_instruct_oddball') is None:
        # initialise key_instruct_oddball
        key_instruct_oddball = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_oddball',
        )
    # create speaker 'read_instruct_oddball'
    deviceManager.addDevice(
        deviceName='read_instruct_oddball',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_instruct_regular') is None:
        # initialise key_instruct_regular
        key_instruct_regular = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_regular',
        )
    # create speaker 'read_instruct_regular'
    deviceManager.addDevice(
        deviceName='read_instruct_regular',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_instruct_distractor') is None:
        # initialise key_instruct_distractor
        key_instruct_distractor = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_distractor',
        )
    # create speaker 'read_instruct_distractor'
    deviceManager.addDevice(
        deviceName='read_instruct_distractor',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_instruct_combined') is None:
        # initialise key_instruct_combined
        key_instruct_combined = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_combined',
        )
    # create speaker 'read_instruct_combined'
    deviceManager.addDevice(
        deviceName='read_instruct_combined',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_instruct_verify') is None:
        # initialise key_instruct_verify
        key_instruct_verify = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_verify',
        )
    # create speaker 'read_instruct_verify'
    deviceManager.addDevice(
        deviceName='read_instruct_verify',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_instruct_begin') is None:
        # initialise key_instruct_begin
        key_instruct_begin = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_begin',
        )
    # create speaker 'read_instruct_begin'
    deviceManager.addDevice(
        deviceName='read_instruct_begin',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('shape_resp') is None:
        # initialise shape_resp
        shape_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='shape_resp',
        )
    # create speaker 'read_thank_you'
    deviceManager.addDevice(
        deviceName='read_thank_you',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "_welcome" ---
    text_welcome = visual.TextStim(win=win, name='text_welcome',
        text='Welcome! This task will take approximately 10 minutes.\n\nPlease take a moment to adjust the chair height, chin rest, and sitting posture. Make sure that you feel comfortable and can stay still for a while.\n\n\nWhen you are ready, press any of the white keys to begin.',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_welcome = keyboard.Keyboard(deviceName='key_welcome')
    read_welcome = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_welcome',    name='read_welcome'
    )
    read_welcome.setVolume(1.0)
    
    # --- Initialize components for Routine "__start__" ---
    # Run 'Begin Experiment' code from trigger_table
    ##TASK ID TRIGGER VALUES##
    # special code 100 (task start, task ID should follow immediately)
    task_start_code = 100
    # special code 111 (task ID for visual double oddball P300 task)
    task_ID_code = 111
    print("Starting experiment: < Visual Double Oddball Task >. Task ID:", task_ID_code)
    
    ##GENERAL TRIGGER VALUES##
    # special code 122 (block start)
    block_start_code = 122
    # special code 123 (block end)
    block_end_code = 123
    
    ##TASK SPECIFIC TRIGGER VALUES##
    # N.B.: only use values 1-99 and provide clear comments on used values
    regular_p300_code = 41
    oddball_p300_code = 42
    distractor_p300_code = 43
    
    # Run 'Begin Experiment' code from task_id
    dev.activate_line(bitmask=task_start_code)  # special code for task start
    core.wait(0.5)  # wait 500ms between two consecutive triggers
    dev.activate_line(bitmask=task_ID_code)  # special code for task ID
    
    # Run 'Begin Experiment' code from condition_setup
    import numpy as np
    
    # --- Stimulus definitions ---
    # (shape, size, type) for each stimulus category
    STIMULI = {
        "regular":   {"edge": 360, "size": 0.3, "text": "small circle"},
        "oddball":    {"edge": 360, "size": 0.5, "text": "big circle"},
        "distractor": {"edge": 4,   "size": 0.6, "text": "rectangle"},
    }
    
    # --- Timing ---
    STIMULUS_DURATION = 0.5
    ITI_RANGE         = (1.5, 2.5)  # seconds
    
    # --- Trial counts ---
    N_CHUNKS = 20  # chunks × 10 trials = 200 trials total
    N_TRIALS = N_CHUNKS * 10  # 160 standard, 20 oddball, 20 distractor per experiment
    
    # --- Build trial sequence ---
    # Each 10-trial chunk:
    #   Position 0 : always standard (no oddball/distractor as first trial)
    #   Positions 1–9 : 7× standard + 1× oddball + 1× distractor, shuffled together
    
    trial_edges = []
    trial_sizes = []
    trial_types = []
    
    for _ in range(N_CHUNKS):
        inner = (
            [("regular",   STIMULI["regular"]["edge"],   STIMULI["regular"]["size"])]   * 7
          + [("oddball",    STIMULI["oddball"]["edge"],    STIMULI["oddball"]["size"])]     * 1
          + [("distractor", STIMULI["distractor"]["edge"], STIMULI["distractor"]["size"])] * 1
        )
        np.random.shuffle(inner)  # shuffles all three fields together — sizes stay aligned
    
        chunk = [("regular", STIMULI["regular"]["edge"], STIMULI["regular"]["size"])] + inner
    
        types, edges, sizes = zip(*chunk)
        trial_types  += list(types)
        trial_edges  += list(edges)
        trial_sizes  += list(sizes)
    
    # --- Verification ---
    assert len(trial_types) == N_TRIALS
    assert trial_types.count("regular")    == 160
    assert trial_types.count("oddball")    == 20
    assert trial_types.count("distractor") == 20
    
    # --- ITI list (includes stimulus duration as onset offset) ---
    iti_list = list(
        ITI_RANGE[0]
        + np.random.random(N_TRIALS) * (ITI_RANGE[1] - ITI_RANGE[0])
        + STIMULUS_DURATION
    )
    
    
    # --- Initialize components for Routine "instruct_oddball" ---
    text_instruct_oddball = visual.TextStim(win=win, name='text_instruct_oddball',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_oddball = keyboard.Keyboard(deviceName='key_instruct_oddball')
    read_instruct_oddball = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_instruct_oddball',    name='read_instruct_oddball'
    )
    read_instruct_oddball.setVolume(1.0)
    
    # --- Initialize components for Routine "shape_oddball" ---
    text_fixation_oddball = visual.TextStim(win=win, name='text_fixation_oddball',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    shape_oddball_ = visual.Polygon(
        win=win, name='shape_oddball_',
        edges=STIMULI["oddball"]["edge"], size=(STIMULI["oddball"]["size"], STIMULI["oddball"]["size"]),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "instruct_regular" ---
    text_instruct_regular = visual.TextStim(win=win, name='text_instruct_regular',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_regular = keyboard.Keyboard(deviceName='key_instruct_regular')
    read_instruct_regular = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_instruct_regular',    name='read_instruct_regular'
    )
    read_instruct_regular.setVolume(1.0)
    
    # --- Initialize components for Routine "shape_regular" ---
    text_fixation_regular = visual.TextStim(win=win, name='text_fixation_regular',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    shape_regular_ = visual.Polygon(
        win=win, name='shape_regular_',
        edges=STIMULI["regular"]["edge"], size=(STIMULI["regular"]["size"], STIMULI["regular"]["size"]),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "instruct_distractor" ---
    text_instruct_distractor = visual.TextStim(win=win, name='text_instruct_distractor',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_distractor = keyboard.Keyboard(deviceName='key_instruct_distractor')
    read_instruct_distractor = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_instruct_distractor',    name='read_instruct_distractor'
    )
    read_instruct_distractor.setVolume(1.0)
    
    # --- Initialize components for Routine "shape_distractor" ---
    text_fixation_distractor = visual.TextStim(win=win, name='text_fixation_distractor',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    shape_distractor_ = visual.Polygon(
        win=win, name='shape_distractor_',
        edges=STIMULI["distractor"]["edge"], size=(STIMULI["distractor"]["size"], STIMULI["distractor"]["size"]),
        ori=45.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "instruct_combined" ---
    text_instruct_combined = visual.TextStim(win=win, name='text_instruct_combined',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_combined = keyboard.Keyboard(deviceName='key_instruct_combined')
    read_instruct_combined = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_instruct_combined',    name='read_instruct_combined'
    )
    read_instruct_combined.setVolume(1.0)
    
    # --- Initialize components for Routine "shape_combined" ---
    text_fixation_combined = visual.TextStim(win=win, name='text_fixation_combined',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    shape_oddball_combined = visual.Polygon(
        win=win, name='shape_oddball_combined',
        edges=STIMULI["oddball"]["edge"], size=(STIMULI["oddball"]["size"], STIMULI["oddball"]["size"]),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    shape_regular_combined = visual.Polygon(
        win=win, name='shape_regular_combined',
        edges=STIMULI["regular"]["edge"], size=(STIMULI["regular"]["size"], STIMULI["regular"]["size"]),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    shape_distractor_combined = visual.Polygon(
        win=win, name='shape_distractor_combined',
        edges=STIMULI["distractor"]["edge"], size=(STIMULI["distractor"]["size"], STIMULI["distractor"]["size"]),
        ori=45.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    
    # --- Initialize components for Routine "instruct_verify" ---
    text_instruct_verify = visual.TextStim(win=win, name='text_instruct_verify',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_verify = keyboard.Keyboard(deviceName='key_instruct_verify')
    read_instruct_verify = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_instruct_verify',    name='read_instruct_verify'
    )
    read_instruct_verify.setVolume(1.0)
    
    # --- Initialize components for Routine "instruct_begin" ---
    text_instruct_begin = visual.TextStim(win=win, name='text_instruct_begin',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_begin = keyboard.Keyboard(deviceName='key_instruct_begin')
    read_instruct_begin = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_instruct_begin',    name='read_instruct_begin'
    )
    read_instruct_begin.setVolume(1.0)
    
    # --- Initialize components for Routine "trial" ---
    stimulus_shape = visual.Polygon(
        win=win, name='stimulus_shape',
        edges=None, size=[1.0, 1.0],
        ori=1.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    text_fixation_cross = visual.TextStim(win=win, name='text_fixation_cross',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    shape_resp = keyboard.Keyboard(deviceName='shape_resp')
    
    # --- Initialize components for Routine "__end__" ---
    text_thank_you = visual.TextStim(win=win, name='text_thank_you',
        text='Thank you. You have completed this task!',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    read_thank_you = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_thank_you',    name='read_thank_you'
    )
    read_thank_you.setVolume(1.0)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "_welcome" ---
    # create an object to store info about Routine _welcome
    _welcome = data.Routine(
        name='_welcome',
        components=[text_welcome, key_welcome, read_welcome],
    )
    _welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_welcome
    key_welcome.keys = []
    key_welcome.rt = []
    _key_welcome_allKeys = []
    read_welcome.setSound('resource/welcome.wav', hamming=True)
    read_welcome.setVolume(1.0, log=False)
    read_welcome.seek(0)
    # store start times for _welcome
    _welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    _welcome.tStart = globalClock.getTime(format='float')
    _welcome.status = STARTED
    _welcome.maxDuration = None
    # keep track of which components have finished
    _welcomeComponents = _welcome.components
    for thisComponent in _welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_welcome" ---
    _welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_welcome* updates
        
        # if text_welcome is starting this frame...
        if text_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_welcome.frameNStart = frameN  # exact frame index
            text_welcome.tStart = t  # local t and not account for scr refresh
            text_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_welcome, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_welcome.status = STARTED
            text_welcome.setAutoDraw(True)
        
        # if text_welcome is active this frame...
        if text_welcome.status == STARTED:
            # update params
            pass
        
        # *key_welcome* updates
        waitOnFlip = False
        
        # if key_welcome is starting this frame...
        if key_welcome.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_welcome.frameNStart = frameN  # exact frame index
            key_welcome.tStart = t  # local t and not account for scr refresh
            key_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_welcome, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_welcome.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_welcome.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_welcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_welcome.status == STARTED and not waitOnFlip:
            theseKeys = key_welcome.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
            _key_welcome_allKeys.extend(theseKeys)
            if len(_key_welcome_allKeys):
                key_welcome.keys = _key_welcome_allKeys[-1].name  # just the last key pressed
                key_welcome.rt = _key_welcome_allKeys[-1].rt
                key_welcome.duration = _key_welcome_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_welcome* updates
        
        # if read_welcome is starting this frame...
        if read_welcome.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_welcome.frameNStart = frameN  # exact frame index
            read_welcome.tStart = t  # local t and not account for scr refresh
            read_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_welcome.status = STARTED
            read_welcome.play(when=win)  # sync with win flip
        
        # if read_welcome is stopping this frame...
        if read_welcome.status == STARTED:
            if bool(False) or read_welcome.isFinished:
                # keep track of stop time/frame for later
                read_welcome.tStop = t  # not accounting for scr refresh
                read_welcome.tStopRefresh = tThisFlipGlobal  # on global time
                read_welcome.frameNStop = frameN  # exact frame index
                # update status
                read_welcome.status = FINISHED
                read_welcome.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_welcome]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            _welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_welcome" ---
    for thisComponent in _welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for _welcome
    _welcome.tStop = globalClock.getTime(format='float')
    _welcome.tStopRefresh = tThisFlipGlobal
    read_welcome.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "_welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "__start__" ---
    # create an object to store info about Routine __start__
    __start__ = data.Routine(
        name='__start__',
        components=[],
    )
    __start__.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for __start__
    __start__.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    __start__.tStart = globalClock.getTime(format='float')
    __start__.status = STARTED
    __start__.maxDuration = None
    # keep track of which components have finished
    __start__Components = __start__.components
    for thisComponent in __start__.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "__start__" ---
    __start__.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            __start__.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in __start__.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "__start__" ---
    for thisComponent in __start__.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for __start__
    __start__.tStop = globalClock.getTime(format='float')
    __start__.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "__start__" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct_oddball" ---
    # create an object to store info about Routine instruct_oddball
    instruct_oddball = data.Routine(
        name='instruct_oddball',
        components=[text_instruct_oddball, key_instruct_oddball, read_instruct_oddball],
    )
    instruct_oddball.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    text_instruct_oddball.setText("INSTRUCTIONS: \n \n In this task you will see a series of shapes. \n \n There are three different shapes. Every time you see the following shape, a "+ STIMULI["oddball"]["text"]+ ", respond by pressing the Green key.\n \nNow press any of the white keys to see the shape.")
    # create starting attributes for key_instruct_oddball
    key_instruct_oddball.keys = []
    key_instruct_oddball.rt = []
    _key_instruct_oddball_allKeys = []
    read_instruct_oddball.setSound('resource/instruct_oddball.wav', hamming=True)
    read_instruct_oddball.setVolume(1.0, log=False)
    read_instruct_oddball.seek(0)
    # store start times for instruct_oddball
    instruct_oddball.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_oddball.tStart = globalClock.getTime(format='float')
    instruct_oddball.status = STARTED
    instruct_oddball.maxDuration = None
    # keep track of which components have finished
    instruct_oddballComponents = instruct_oddball.components
    for thisComponent in instruct_oddball.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_oddball" ---
    instruct_oddball.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruct_oddball* updates
        
        # if text_instruct_oddball is starting this frame...
        if text_instruct_oddball.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruct_oddball.frameNStart = frameN  # exact frame index
            text_instruct_oddball.tStart = t  # local t and not account for scr refresh
            text_instruct_oddball.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruct_oddball, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instruct_oddball.status = STARTED
            text_instruct_oddball.setAutoDraw(True)
        
        # if text_instruct_oddball is active this frame...
        if text_instruct_oddball.status == STARTED:
            # update params
            pass
        
        # *key_instruct_oddball* updates
        waitOnFlip = False
        
        # if key_instruct_oddball is starting this frame...
        if key_instruct_oddball.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_oddball.frameNStart = frameN  # exact frame index
            key_instruct_oddball.tStart = t  # local t and not account for scr refresh
            key_instruct_oddball.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_oddball, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_instruct_oddball.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_oddball.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_oddball.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_oddball.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_oddball.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
            _key_instruct_oddball_allKeys.extend(theseKeys)
            if len(_key_instruct_oddball_allKeys):
                key_instruct_oddball.keys = _key_instruct_oddball_allKeys[-1].name  # just the last key pressed
                key_instruct_oddball.rt = _key_instruct_oddball_allKeys[-1].rt
                key_instruct_oddball.duration = _key_instruct_oddball_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_instruct_oddball* updates
        
        # if read_instruct_oddball is starting this frame...
        if read_instruct_oddball.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_instruct_oddball.frameNStart = frameN  # exact frame index
            read_instruct_oddball.tStart = t  # local t and not account for scr refresh
            read_instruct_oddball.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_instruct_oddball.status = STARTED
            read_instruct_oddball.play(when=win)  # sync with win flip
        
        # if read_instruct_oddball is stopping this frame...
        if read_instruct_oddball.status == STARTED:
            if bool(False) or read_instruct_oddball.isFinished:
                # keep track of stop time/frame for later
                read_instruct_oddball.tStop = t  # not accounting for scr refresh
                read_instruct_oddball.tStopRefresh = tThisFlipGlobal  # on global time
                read_instruct_oddball.frameNStop = frameN  # exact frame index
                # update status
                read_instruct_oddball.status = FINISHED
                read_instruct_oddball.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_instruct_oddball]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instruct_oddball.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_oddball.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_oddball" ---
    for thisComponent in instruct_oddball.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_oddball
    instruct_oddball.tStop = globalClock.getTime(format='float')
    instruct_oddball.tStopRefresh = tThisFlipGlobal
    read_instruct_oddball.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instruct_oddball" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "shape_oddball" ---
    # create an object to store info about Routine shape_oddball
    shape_oddball = data.Routine(
        name='shape_oddball',
        components=[text_fixation_oddball, shape_oddball_],
    )
    shape_oddball.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for shape_oddball
    shape_oddball.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    shape_oddball.tStart = globalClock.getTime(format='float')
    shape_oddball.status = STARTED
    shape_oddball.maxDuration = None
    # keep track of which components have finished
    shape_oddballComponents = shape_oddball.components
    for thisComponent in shape_oddball.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "shape_oddball" ---
    shape_oddball.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_fixation_oddball* updates
        
        # if text_fixation_oddball is starting this frame...
        if text_fixation_oddball.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_fixation_oddball.frameNStart = frameN  # exact frame index
            text_fixation_oddball.tStart = t  # local t and not account for scr refresh
            text_fixation_oddball.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_fixation_oddball, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_fixation_oddball.status = STARTED
            text_fixation_oddball.setAutoDraw(True)
        
        # if text_fixation_oddball is active this frame...
        if text_fixation_oddball.status == STARTED:
            # update params
            pass
        
        # if text_fixation_oddball is stopping this frame...
        if text_fixation_oddball.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_fixation_oddball.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                text_fixation_oddball.tStop = t  # not accounting for scr refresh
                text_fixation_oddball.tStopRefresh = tThisFlipGlobal  # on global time
                text_fixation_oddball.frameNStop = frameN  # exact frame index
                # update status
                text_fixation_oddball.status = FINISHED
                text_fixation_oddball.setAutoDraw(False)
        
        # *shape_oddball_* updates
        
        # if shape_oddball_ is starting this frame...
        if shape_oddball_.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            shape_oddball_.frameNStart = frameN  # exact frame index
            shape_oddball_.tStart = t  # local t and not account for scr refresh
            shape_oddball_.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(shape_oddball_, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'shape_oddball_.started')
            # update status
            shape_oddball_.status = STARTED
            shape_oddball_.setAutoDraw(True)
        
        # if shape_oddball_ is active this frame...
        if shape_oddball_.status == STARTED:
            # update params
            pass
        
        # if shape_oddball_ is stopping this frame...
        if shape_oddball_.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > shape_oddball_.tStartRefresh + STIMULUS_DURATION-frameTolerance:
                # keep track of stop time/frame for later
                shape_oddball_.tStop = t  # not accounting for scr refresh
                shape_oddball_.tStopRefresh = tThisFlipGlobal  # on global time
                shape_oddball_.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'shape_oddball_.stopped')
                # update status
                shape_oddball_.status = FINISHED
                shape_oddball_.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            shape_oddball.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in shape_oddball.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "shape_oddball" ---
    for thisComponent in shape_oddball.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for shape_oddball
    shape_oddball.tStop = globalClock.getTime(format='float')
    shape_oddball.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "shape_oddball" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct_regular" ---
    # create an object to store info about Routine instruct_regular
    instruct_regular = data.Routine(
        name='instruct_regular',
        components=[text_instruct_regular, key_instruct_regular, read_instruct_regular],
    )
    instruct_regular.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    text_instruct_regular.setText("INSTRUCTIONS:\n \n Every time you see the following shape, a " + STIMULI["regular"]["text"] + ", don't press any key. \n \n You do not need to make any response. \n \n Now press any of the white keys to see the shape.")
    # create starting attributes for key_instruct_regular
    key_instruct_regular.keys = []
    key_instruct_regular.rt = []
    _key_instruct_regular_allKeys = []
    read_instruct_regular.setSound('resource/instruct_regular.wav', hamming=True)
    read_instruct_regular.setVolume(1.0, log=False)
    read_instruct_regular.seek(0)
    # store start times for instruct_regular
    instruct_regular.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_regular.tStart = globalClock.getTime(format='float')
    instruct_regular.status = STARTED
    instruct_regular.maxDuration = None
    # keep track of which components have finished
    instruct_regularComponents = instruct_regular.components
    for thisComponent in instruct_regular.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_regular" ---
    instruct_regular.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruct_regular* updates
        
        # if text_instruct_regular is starting this frame...
        if text_instruct_regular.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruct_regular.frameNStart = frameN  # exact frame index
            text_instruct_regular.tStart = t  # local t and not account for scr refresh
            text_instruct_regular.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruct_regular, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instruct_regular.status = STARTED
            text_instruct_regular.setAutoDraw(True)
        
        # if text_instruct_regular is active this frame...
        if text_instruct_regular.status == STARTED:
            # update params
            pass
        
        # *key_instruct_regular* updates
        waitOnFlip = False
        
        # if key_instruct_regular is starting this frame...
        if key_instruct_regular.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_regular.frameNStart = frameN  # exact frame index
            key_instruct_regular.tStart = t  # local t and not account for scr refresh
            key_instruct_regular.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_regular, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_instruct_regular.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_regular.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_regular.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_regular.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_regular.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
            _key_instruct_regular_allKeys.extend(theseKeys)
            if len(_key_instruct_regular_allKeys):
                key_instruct_regular.keys = _key_instruct_regular_allKeys[-1].name  # just the last key pressed
                key_instruct_regular.rt = _key_instruct_regular_allKeys[-1].rt
                key_instruct_regular.duration = _key_instruct_regular_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_instruct_regular* updates
        
        # if read_instruct_regular is starting this frame...
        if read_instruct_regular.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_instruct_regular.frameNStart = frameN  # exact frame index
            read_instruct_regular.tStart = t  # local t and not account for scr refresh
            read_instruct_regular.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_instruct_regular.status = STARTED
            read_instruct_regular.play(when=win)  # sync with win flip
        
        # if read_instruct_regular is stopping this frame...
        if read_instruct_regular.status == STARTED:
            if bool(False) or read_instruct_regular.isFinished:
                # keep track of stop time/frame for later
                read_instruct_regular.tStop = t  # not accounting for scr refresh
                read_instruct_regular.tStopRefresh = tThisFlipGlobal  # on global time
                read_instruct_regular.frameNStop = frameN  # exact frame index
                # update status
                read_instruct_regular.status = FINISHED
                read_instruct_regular.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_instruct_regular]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instruct_regular.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_regular.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_regular" ---
    for thisComponent in instruct_regular.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_regular
    instruct_regular.tStop = globalClock.getTime(format='float')
    instruct_regular.tStopRefresh = tThisFlipGlobal
    read_instruct_regular.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instruct_regular" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "shape_regular" ---
    # create an object to store info about Routine shape_regular
    shape_regular = data.Routine(
        name='shape_regular',
        components=[text_fixation_regular, shape_regular_],
    )
    shape_regular.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for shape_regular
    shape_regular.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    shape_regular.tStart = globalClock.getTime(format='float')
    shape_regular.status = STARTED
    shape_regular.maxDuration = None
    # keep track of which components have finished
    shape_regularComponents = shape_regular.components
    for thisComponent in shape_regular.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "shape_regular" ---
    shape_regular.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_fixation_regular* updates
        
        # if text_fixation_regular is starting this frame...
        if text_fixation_regular.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_fixation_regular.frameNStart = frameN  # exact frame index
            text_fixation_regular.tStart = t  # local t and not account for scr refresh
            text_fixation_regular.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_fixation_regular, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_fixation_regular.status = STARTED
            text_fixation_regular.setAutoDraw(True)
        
        # if text_fixation_regular is active this frame...
        if text_fixation_regular.status == STARTED:
            # update params
            pass
        
        # if text_fixation_regular is stopping this frame...
        if text_fixation_regular.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_fixation_regular.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                text_fixation_regular.tStop = t  # not accounting for scr refresh
                text_fixation_regular.tStopRefresh = tThisFlipGlobal  # on global time
                text_fixation_regular.frameNStop = frameN  # exact frame index
                # update status
                text_fixation_regular.status = FINISHED
                text_fixation_regular.setAutoDraw(False)
        
        # *shape_regular_* updates
        
        # if shape_regular_ is starting this frame...
        if shape_regular_.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            shape_regular_.frameNStart = frameN  # exact frame index
            shape_regular_.tStart = t  # local t and not account for scr refresh
            shape_regular_.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(shape_regular_, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'shape_regular_.started')
            # update status
            shape_regular_.status = STARTED
            shape_regular_.setAutoDraw(True)
        
        # if shape_regular_ is active this frame...
        if shape_regular_.status == STARTED:
            # update params
            pass
        
        # if shape_regular_ is stopping this frame...
        if shape_regular_.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > shape_regular_.tStartRefresh + STIMULUS_DURATION-frameTolerance:
                # keep track of stop time/frame for later
                shape_regular_.tStop = t  # not accounting for scr refresh
                shape_regular_.tStopRefresh = tThisFlipGlobal  # on global time
                shape_regular_.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'shape_regular_.stopped')
                # update status
                shape_regular_.status = FINISHED
                shape_regular_.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            shape_regular.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in shape_regular.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "shape_regular" ---
    for thisComponent in shape_regular.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for shape_regular
    shape_regular.tStop = globalClock.getTime(format='float')
    shape_regular.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "shape_regular" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct_distractor" ---
    # create an object to store info about Routine instruct_distractor
    instruct_distractor = data.Routine(
        name='instruct_distractor',
        components=[text_instruct_distractor, key_instruct_distractor, read_instruct_distractor],
    )
    instruct_distractor.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    text_instruct_distractor.setText("INSTRUCTIONS:\n \n Every time you see the following shape, a " + STIMULI["distractor"]["text"] + ", don't press any key. \n \n You do not need to make any response. \n \n Now press any of the white keys to see the shape.")
    # create starting attributes for key_instruct_distractor
    key_instruct_distractor.keys = []
    key_instruct_distractor.rt = []
    _key_instruct_distractor_allKeys = []
    read_instruct_distractor.setSound('resource/instruct_distractor.wav', hamming=True)
    read_instruct_distractor.setVolume(1.0, log=False)
    read_instruct_distractor.seek(0)
    # store start times for instruct_distractor
    instruct_distractor.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_distractor.tStart = globalClock.getTime(format='float')
    instruct_distractor.status = STARTED
    instruct_distractor.maxDuration = None
    # keep track of which components have finished
    instruct_distractorComponents = instruct_distractor.components
    for thisComponent in instruct_distractor.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_distractor" ---
    instruct_distractor.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruct_distractor* updates
        
        # if text_instruct_distractor is starting this frame...
        if text_instruct_distractor.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruct_distractor.frameNStart = frameN  # exact frame index
            text_instruct_distractor.tStart = t  # local t and not account for scr refresh
            text_instruct_distractor.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruct_distractor, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instruct_distractor.status = STARTED
            text_instruct_distractor.setAutoDraw(True)
        
        # if text_instruct_distractor is active this frame...
        if text_instruct_distractor.status == STARTED:
            # update params
            pass
        
        # *key_instruct_distractor* updates
        waitOnFlip = False
        
        # if key_instruct_distractor is starting this frame...
        if key_instruct_distractor.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_distractor.frameNStart = frameN  # exact frame index
            key_instruct_distractor.tStart = t  # local t and not account for scr refresh
            key_instruct_distractor.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_distractor, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_instruct_distractor.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_distractor.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_distractor.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_distractor.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_distractor.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
            _key_instruct_distractor_allKeys.extend(theseKeys)
            if len(_key_instruct_distractor_allKeys):
                key_instruct_distractor.keys = _key_instruct_distractor_allKeys[-1].name  # just the last key pressed
                key_instruct_distractor.rt = _key_instruct_distractor_allKeys[-1].rt
                key_instruct_distractor.duration = _key_instruct_distractor_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_instruct_distractor* updates
        
        # if read_instruct_distractor is starting this frame...
        if read_instruct_distractor.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_instruct_distractor.frameNStart = frameN  # exact frame index
            read_instruct_distractor.tStart = t  # local t and not account for scr refresh
            read_instruct_distractor.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_instruct_distractor.status = STARTED
            read_instruct_distractor.play(when=win)  # sync with win flip
        
        # if read_instruct_distractor is stopping this frame...
        if read_instruct_distractor.status == STARTED:
            if bool(False) or read_instruct_distractor.isFinished:
                # keep track of stop time/frame for later
                read_instruct_distractor.tStop = t  # not accounting for scr refresh
                read_instruct_distractor.tStopRefresh = tThisFlipGlobal  # on global time
                read_instruct_distractor.frameNStop = frameN  # exact frame index
                # update status
                read_instruct_distractor.status = FINISHED
                read_instruct_distractor.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_instruct_distractor]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instruct_distractor.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_distractor.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_distractor" ---
    for thisComponent in instruct_distractor.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_distractor
    instruct_distractor.tStop = globalClock.getTime(format='float')
    instruct_distractor.tStopRefresh = tThisFlipGlobal
    read_instruct_distractor.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instruct_distractor" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "shape_distractor" ---
    # create an object to store info about Routine shape_distractor
    shape_distractor = data.Routine(
        name='shape_distractor',
        components=[text_fixation_distractor, shape_distractor_],
    )
    shape_distractor.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for shape_distractor
    shape_distractor.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    shape_distractor.tStart = globalClock.getTime(format='float')
    shape_distractor.status = STARTED
    shape_distractor.maxDuration = None
    # keep track of which components have finished
    shape_distractorComponents = shape_distractor.components
    for thisComponent in shape_distractor.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "shape_distractor" ---
    shape_distractor.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_fixation_distractor* updates
        
        # if text_fixation_distractor is starting this frame...
        if text_fixation_distractor.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_fixation_distractor.frameNStart = frameN  # exact frame index
            text_fixation_distractor.tStart = t  # local t and not account for scr refresh
            text_fixation_distractor.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_fixation_distractor, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_fixation_distractor.status = STARTED
            text_fixation_distractor.setAutoDraw(True)
        
        # if text_fixation_distractor is active this frame...
        if text_fixation_distractor.status == STARTED:
            # update params
            pass
        
        # if text_fixation_distractor is stopping this frame...
        if text_fixation_distractor.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_fixation_distractor.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                text_fixation_distractor.tStop = t  # not accounting for scr refresh
                text_fixation_distractor.tStopRefresh = tThisFlipGlobal  # on global time
                text_fixation_distractor.frameNStop = frameN  # exact frame index
                # update status
                text_fixation_distractor.status = FINISHED
                text_fixation_distractor.setAutoDraw(False)
        
        # *shape_distractor_* updates
        
        # if shape_distractor_ is starting this frame...
        if shape_distractor_.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            shape_distractor_.frameNStart = frameN  # exact frame index
            shape_distractor_.tStart = t  # local t and not account for scr refresh
            shape_distractor_.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(shape_distractor_, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'shape_distractor_.started')
            # update status
            shape_distractor_.status = STARTED
            shape_distractor_.setAutoDraw(True)
        
        # if shape_distractor_ is active this frame...
        if shape_distractor_.status == STARTED:
            # update params
            pass
        
        # if shape_distractor_ is stopping this frame...
        if shape_distractor_.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > shape_distractor_.tStartRefresh + STIMULUS_DURATION-frameTolerance:
                # keep track of stop time/frame for later
                shape_distractor_.tStop = t  # not accounting for scr refresh
                shape_distractor_.tStopRefresh = tThisFlipGlobal  # on global time
                shape_distractor_.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'shape_distractor_.stopped')
                # update status
                shape_distractor_.status = FINISHED
                shape_distractor_.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            shape_distractor.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in shape_distractor.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "shape_distractor" ---
    for thisComponent in shape_distractor.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for shape_distractor
    shape_distractor.tStop = globalClock.getTime(format='float')
    shape_distractor.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "shape_distractor" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_loop = data.TrialHandler2(
        name='practice_loop',
        nReps=99.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(practice_loop)  # add the loop to the experiment
    thisPractice_loop = practice_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
    if thisPractice_loop != None:
        for paramName in thisPractice_loop:
            globals()[paramName] = thisPractice_loop[paramName]
    
    for thisPractice_loop in practice_loop:
        currentLoop = practice_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
        if thisPractice_loop != None:
            for paramName in thisPractice_loop:
                globals()[paramName] = thisPractice_loop[paramName]
        
        # --- Prepare to start Routine "instruct_combined" ---
        # create an object to store info about Routine instruct_combined
        instruct_combined = data.Routine(
            name='instruct_combined',
            components=[text_instruct_combined, key_instruct_combined, read_instruct_combined],
        )
        instruct_combined.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_instruct_combined.setText("INSTRUCTIONS: Now we will see all the three shapes again.\n \nPay attention and decide which shape you need to respond to:\n \nis it the " + STIMULI['oddball']['text'] + ", the " + STIMULI['regular']['text'] + ", or the " + STIMULI['distractor']['text'] + "?\n \nPress any of the white keys to see the three shapes together.")
        # create starting attributes for key_instruct_combined
        key_instruct_combined.keys = []
        key_instruct_combined.rt = []
        _key_instruct_combined_allKeys = []
        read_instruct_combined.setSound('resource/instruct_combined.wav', hamming=True)
        read_instruct_combined.setVolume(1.0, log=False)
        read_instruct_combined.seek(0)
        # store start times for instruct_combined
        instruct_combined.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        instruct_combined.tStart = globalClock.getTime(format='float')
        instruct_combined.status = STARTED
        instruct_combined.maxDuration = None
        # keep track of which components have finished
        instruct_combinedComponents = instruct_combined.components
        for thisComponent in instruct_combined.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "instruct_combined" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop, data.TrialHandler2) and thisPractice_loop.thisN != practice_loop.thisTrial.thisN:
            continueRoutine = False
        instruct_combined.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_instruct_combined* updates
            
            # if text_instruct_combined is starting this frame...
            if text_instruct_combined.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_instruct_combined.frameNStart = frameN  # exact frame index
                text_instruct_combined.tStart = t  # local t and not account for scr refresh
                text_instruct_combined.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_instruct_combined, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_instruct_combined.status = STARTED
                text_instruct_combined.setAutoDraw(True)
            
            # if text_instruct_combined is active this frame...
            if text_instruct_combined.status == STARTED:
                # update params
                pass
            
            # *key_instruct_combined* updates
            waitOnFlip = False
            
            # if key_instruct_combined is starting this frame...
            if key_instruct_combined.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                key_instruct_combined.frameNStart = frameN  # exact frame index
                key_instruct_combined.tStart = t  # local t and not account for scr refresh
                key_instruct_combined.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_instruct_combined, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_instruct_combined.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_instruct_combined.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_instruct_combined.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_instruct_combined.status == STARTED and not waitOnFlip:
                theseKeys = key_instruct_combined.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
                _key_instruct_combined_allKeys.extend(theseKeys)
                if len(_key_instruct_combined_allKeys):
                    key_instruct_combined.keys = _key_instruct_combined_allKeys[-1].name  # just the last key pressed
                    key_instruct_combined.rt = _key_instruct_combined_allKeys[-1].rt
                    key_instruct_combined.duration = _key_instruct_combined_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *read_instruct_combined* updates
            
            # if read_instruct_combined is starting this frame...
            if read_instruct_combined.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
                # keep track of start time/frame for later
                read_instruct_combined.frameNStart = frameN  # exact frame index
                read_instruct_combined.tStart = t  # local t and not account for scr refresh
                read_instruct_combined.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                read_instruct_combined.status = STARTED
                read_instruct_combined.play(when=win)  # sync with win flip
            
            # if read_instruct_combined is stopping this frame...
            if read_instruct_combined.status == STARTED:
                if bool(False) or read_instruct_combined.isFinished:
                    # keep track of stop time/frame for later
                    read_instruct_combined.tStop = t  # not accounting for scr refresh
                    read_instruct_combined.tStopRefresh = tThisFlipGlobal  # on global time
                    read_instruct_combined.frameNStop = frameN  # exact frame index
                    # update status
                    read_instruct_combined.status = FINISHED
                    read_instruct_combined.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[read_instruct_combined]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                instruct_combined.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instruct_combined.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instruct_combined" ---
        for thisComponent in instruct_combined.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for instruct_combined
        instruct_combined.tStop = globalClock.getTime(format='float')
        instruct_combined.tStopRefresh = tThisFlipGlobal
        read_instruct_combined.pause()  # ensure sound has stopped at end of Routine
        # the Routine "instruct_combined" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "shape_combined" ---
        # create an object to store info about Routine shape_combined
        shape_combined = data.Routine(
            name='shape_combined',
            components=[text_fixation_combined, shape_oddball_combined, shape_regular_combined, shape_distractor_combined],
        )
        shape_combined.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for shape_combined
        shape_combined.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        shape_combined.tStart = globalClock.getTime(format='float')
        shape_combined.status = STARTED
        shape_combined.maxDuration = None
        # keep track of which components have finished
        shape_combinedComponents = shape_combined.components
        for thisComponent in shape_combined.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "shape_combined" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop, data.TrialHandler2) and thisPractice_loop.thisN != practice_loop.thisTrial.thisN:
            continueRoutine = False
        shape_combined.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_fixation_combined* updates
            
            # if text_fixation_combined is starting this frame...
            if text_fixation_combined.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_fixation_combined.frameNStart = frameN  # exact frame index
                text_fixation_combined.tStart = t  # local t and not account for scr refresh
                text_fixation_combined.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_fixation_combined, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_fixation_combined.status = STARTED
                text_fixation_combined.setAutoDraw(True)
            
            # if text_fixation_combined is active this frame...
            if text_fixation_combined.status == STARTED:
                # update params
                pass
            
            # if text_fixation_combined is stopping this frame...
            if text_fixation_combined.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_fixation_combined.tStartRefresh + 3.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_fixation_combined.tStop = t  # not accounting for scr refresh
                    text_fixation_combined.tStopRefresh = tThisFlipGlobal  # on global time
                    text_fixation_combined.frameNStop = frameN  # exact frame index
                    # update status
                    text_fixation_combined.status = FINISHED
                    text_fixation_combined.setAutoDraw(False)
            
            # *shape_oddball_combined* updates
            
            # if shape_oddball_combined is starting this frame...
            if shape_oddball_combined.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                shape_oddball_combined.frameNStart = frameN  # exact frame index
                shape_oddball_combined.tStart = t  # local t and not account for scr refresh
                shape_oddball_combined.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(shape_oddball_combined, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'shape_oddball_combined.started')
                # update status
                shape_oddball_combined.status = STARTED
                shape_oddball_combined.setAutoDraw(True)
            
            # if shape_oddball_combined is active this frame...
            if shape_oddball_combined.status == STARTED:
                # update params
                pass
            
            # if shape_oddball_combined is stopping this frame...
            if shape_oddball_combined.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > shape_oddball_combined.tStartRefresh + STIMULUS_DURATION-frameTolerance:
                    # keep track of stop time/frame for later
                    shape_oddball_combined.tStop = t  # not accounting for scr refresh
                    shape_oddball_combined.tStopRefresh = tThisFlipGlobal  # on global time
                    shape_oddball_combined.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'shape_oddball_combined.stopped')
                    # update status
                    shape_oddball_combined.status = FINISHED
                    shape_oddball_combined.setAutoDraw(False)
            
            # *shape_regular_combined* updates
            
            # if shape_regular_combined is starting this frame...
            if shape_regular_combined.status == NOT_STARTED and tThisFlip >= 1.7-frameTolerance:
                # keep track of start time/frame for later
                shape_regular_combined.frameNStart = frameN  # exact frame index
                shape_regular_combined.tStart = t  # local t and not account for scr refresh
                shape_regular_combined.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(shape_regular_combined, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'shape_regular_combined.started')
                # update status
                shape_regular_combined.status = STARTED
                shape_regular_combined.setAutoDraw(True)
            
            # if shape_regular_combined is active this frame...
            if shape_regular_combined.status == STARTED:
                # update params
                pass
            
            # if shape_regular_combined is stopping this frame...
            if shape_regular_combined.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > shape_regular_combined.tStartRefresh + STIMULUS_DURATION-frameTolerance:
                    # keep track of stop time/frame for later
                    shape_regular_combined.tStop = t  # not accounting for scr refresh
                    shape_regular_combined.tStopRefresh = tThisFlipGlobal  # on global time
                    shape_regular_combined.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'shape_regular_combined.stopped')
                    # update status
                    shape_regular_combined.status = FINISHED
                    shape_regular_combined.setAutoDraw(False)
            
            # *shape_distractor_combined* updates
            
            # if shape_distractor_combined is starting this frame...
            if shape_distractor_combined.status == NOT_STARTED and tThisFlip >= 2.9-frameTolerance:
                # keep track of start time/frame for later
                shape_distractor_combined.frameNStart = frameN  # exact frame index
                shape_distractor_combined.tStart = t  # local t and not account for scr refresh
                shape_distractor_combined.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(shape_distractor_combined, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'shape_distractor_combined.started')
                # update status
                shape_distractor_combined.status = STARTED
                shape_distractor_combined.setAutoDraw(True)
            
            # if shape_distractor_combined is active this frame...
            if shape_distractor_combined.status == STARTED:
                # update params
                pass
            
            # if shape_distractor_combined is stopping this frame...
            if shape_distractor_combined.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > shape_distractor_combined.tStartRefresh + STIMULUS_DURATION-frameTolerance:
                    # keep track of stop time/frame for later
                    shape_distractor_combined.tStop = t  # not accounting for scr refresh
                    shape_distractor_combined.tStopRefresh = tThisFlipGlobal  # on global time
                    shape_distractor_combined.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'shape_distractor_combined.stopped')
                    # update status
                    shape_distractor_combined.status = FINISHED
                    shape_distractor_combined.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                shape_combined.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in shape_combined.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "shape_combined" ---
        for thisComponent in shape_combined.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for shape_combined
        shape_combined.tStop = globalClock.getTime(format='float')
        shape_combined.tStopRefresh = tThisFlipGlobal
        # the Routine "shape_combined" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "instruct_verify" ---
        # create an object to store info about Routine instruct_verify
        instruct_verify = data.Routine(
            name='instruct_verify',
            components=[text_instruct_verify, key_instruct_verify, read_instruct_verify],
        )
        instruct_verify.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_instruct_verify.setText("Would you respond to the " + STIMULI['oddball']['text'] + ", the " + STIMULI['regular']['text'] + ", or the " + STIMULI['distractor']['text'] + "? \n \nPlease tell the examiner verbally. You can also ask to see the shapes again if you are not sure.")
        # create starting attributes for key_instruct_verify
        key_instruct_verify.keys = []
        key_instruct_verify.rt = []
        _key_instruct_verify_allKeys = []
        read_instruct_verify.setSound('resource/instruct_verify.wav', hamming=True)
        read_instruct_verify.setVolume(1.0, log=False)
        read_instruct_verify.seek(0)
        # store start times for instruct_verify
        instruct_verify.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        instruct_verify.tStart = globalClock.getTime(format='float')
        instruct_verify.status = STARTED
        instruct_verify.maxDuration = None
        # keep track of which components have finished
        instruct_verifyComponents = instruct_verify.components
        for thisComponent in instruct_verify.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "instruct_verify" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop, data.TrialHandler2) and thisPractice_loop.thisN != practice_loop.thisTrial.thisN:
            continueRoutine = False
        instruct_verify.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_instruct_verify* updates
            
            # if text_instruct_verify is starting this frame...
            if text_instruct_verify.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_instruct_verify.frameNStart = frameN  # exact frame index
                text_instruct_verify.tStart = t  # local t and not account for scr refresh
                text_instruct_verify.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_instruct_verify, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_instruct_verify.status = STARTED
                text_instruct_verify.setAutoDraw(True)
            
            # if text_instruct_verify is active this frame...
            if text_instruct_verify.status == STARTED:
                # update params
                pass
            
            # *key_instruct_verify* updates
            waitOnFlip = False
            
            # if key_instruct_verify is starting this frame...
            if key_instruct_verify.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                key_instruct_verify.frameNStart = frameN  # exact frame index
                key_instruct_verify.tStart = t  # local t and not account for scr refresh
                key_instruct_verify.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_instruct_verify, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_instruct_verify.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_instruct_verify.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_instruct_verify.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_instruct_verify.status == STARTED and not waitOnFlip:
                theseKeys = key_instruct_verify.getKeys(keyList=['r', 'o'], ignoreKeys=["escape"], waitRelease=True)
                _key_instruct_verify_allKeys.extend(theseKeys)
                if len(_key_instruct_verify_allKeys):
                    key_instruct_verify.keys = _key_instruct_verify_allKeys[-1].name  # just the last key pressed
                    key_instruct_verify.rt = _key_instruct_verify_allKeys[-1].rt
                    key_instruct_verify.duration = _key_instruct_verify_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *read_instruct_verify* updates
            
            # if read_instruct_verify is starting this frame...
            if read_instruct_verify.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
                # keep track of start time/frame for later
                read_instruct_verify.frameNStart = frameN  # exact frame index
                read_instruct_verify.tStart = t  # local t and not account for scr refresh
                read_instruct_verify.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                read_instruct_verify.status = STARTED
                read_instruct_verify.play(when=win)  # sync with win flip
            
            # if read_instruct_verify is stopping this frame...
            if read_instruct_verify.status == STARTED:
                if bool(False) or read_instruct_verify.isFinished:
                    # keep track of stop time/frame for later
                    read_instruct_verify.tStop = t  # not accounting for scr refresh
                    read_instruct_verify.tStopRefresh = tThisFlipGlobal  # on global time
                    read_instruct_verify.frameNStop = frameN  # exact frame index
                    # update status
                    read_instruct_verify.status = FINISHED
                    read_instruct_verify.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[read_instruct_verify]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                instruct_verify.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instruct_verify.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instruct_verify" ---
        for thisComponent in instruct_verify.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for instruct_verify
        instruct_verify.tStop = globalClock.getTime(format='float')
        instruct_verify.tStopRefresh = tThisFlipGlobal
        read_instruct_verify.pause()  # ensure sound has stopped at end of Routine
        # Run 'End Routine' code from code_instruct_verify
        if key_instruct_verify.keys == 'o':  # proceed to main experiment
            practice_loop.finished = True
        
        # the Routine "instruct_verify" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 99.0 repeats of 'practice_loop'
    
    
    # --- Prepare to start Routine "instruct_begin" ---
    # create an object to store info about Routine instruct_begin
    instruct_begin = data.Routine(
        name='instruct_begin',
        components=[text_instruct_begin, key_instruct_begin, read_instruct_begin],
    )
    instruct_begin.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    text_instruct_begin.setText("INSTRUCTIONS:\n \nRemember to press the Green key every time you see the " + STIMULI['oddball']['text'] + ", and otherwise do not respond. Please try to respond as quickly and accurately as possible.\n \nPress the green key to begin the task.")
    # create starting attributes for key_instruct_begin
    key_instruct_begin.keys = []
    key_instruct_begin.rt = []
    _key_instruct_begin_allKeys = []
    read_instruct_begin.setSound('resource/instruct_begin.wav', hamming=True)
    read_instruct_begin.setVolume(1.0, log=False)
    read_instruct_begin.seek(0)
    # Run 'Begin Routine' code from trigger_trial_block
    # Beginning of main experiment trial block
    dev.activate_line(bitmask=block_start_code)
    # no need to wait 500ms because this routine lasts 2.0s before trial triggers
    
    # store start times for instruct_begin
    instruct_begin.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_begin.tStart = globalClock.getTime(format='float')
    instruct_begin.status = STARTED
    instruct_begin.maxDuration = None
    # keep track of which components have finished
    instruct_beginComponents = instruct_begin.components
    for thisComponent in instruct_begin.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_begin" ---
    instruct_begin.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruct_begin* updates
        
        # if text_instruct_begin is starting this frame...
        if text_instruct_begin.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruct_begin.frameNStart = frameN  # exact frame index
            text_instruct_begin.tStart = t  # local t and not account for scr refresh
            text_instruct_begin.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruct_begin, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instruct_begin.status = STARTED
            text_instruct_begin.setAutoDraw(True)
        
        # if text_instruct_begin is active this frame...
        if text_instruct_begin.status == STARTED:
            # update params
            pass
        
        # *key_instruct_begin* updates
        waitOnFlip = False
        
        # if key_instruct_begin is starting this frame...
        if key_instruct_begin.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_begin.frameNStart = frameN  # exact frame index
            key_instruct_begin.tStart = t  # local t and not account for scr refresh
            key_instruct_begin.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_begin, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_instruct_begin.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_begin.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_begin.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_begin.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_begin.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=True)
            _key_instruct_begin_allKeys.extend(theseKeys)
            if len(_key_instruct_begin_allKeys):
                key_instruct_begin.keys = _key_instruct_begin_allKeys[-1].name  # just the last key pressed
                key_instruct_begin.rt = _key_instruct_begin_allKeys[-1].rt
                key_instruct_begin.duration = _key_instruct_begin_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *read_instruct_begin* updates
        
        # if read_instruct_begin is starting this frame...
        if read_instruct_begin.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_instruct_begin.frameNStart = frameN  # exact frame index
            read_instruct_begin.tStart = t  # local t and not account for scr refresh
            read_instruct_begin.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_instruct_begin.status = STARTED
            read_instruct_begin.play(when=win)  # sync with win flip
        
        # if read_instruct_begin is stopping this frame...
        if read_instruct_begin.status == STARTED:
            if bool(False) or read_instruct_begin.isFinished:
                # keep track of stop time/frame for later
                read_instruct_begin.tStop = t  # not accounting for scr refresh
                read_instruct_begin.tStopRefresh = tThisFlipGlobal  # on global time
                read_instruct_begin.frameNStop = frameN  # exact frame index
                # update status
                read_instruct_begin.status = FINISHED
                read_instruct_begin.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_instruct_begin]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instruct_begin.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_begin.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_begin" ---
    for thisComponent in instruct_begin.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_begin
    instruct_begin.tStop = globalClock.getTime(format='float')
    instruct_begin.tStopRefresh = tThisFlipGlobal
    read_instruct_begin.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instruct_begin" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=N_TRIALS, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "trial" ---
        # create an object to store info about Routine trial
        trial = data.Routine(
            name='trial',
            components=[stimulus_shape, text_fixation_cross, shape_resp],
        )
        trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from set_shape_edges
        # Pop the next trial's properties from the lists
        iti          = iti_list.pop(0)
        shape_edges  = trial_edges.pop(0)
        shape_size   = trial_sizes.pop(0)
        trial_type   = trial_types.pop(0)
        
        # Derive oddball/distractor flags from trial_type
        shape_is_oddball    = trial_type == "oddball"
        shape_is_distractor = trial_type == "distractor"
        
        # Log to data table
        thisExp.addData('trial_type',          trial_type)
        thisExp.addData('shape_is_oddball',    shape_is_oddball)
        thisExp.addData('shape_is_distractor', shape_is_distractor)
        thisExp.addData('iti',                 iti - STIMULUS_DURATION)
        
        stimulus_shape.setSize((shape_size, shape_size))
        stimulus_shape.setOri(45 if trial_type == "distractor" else 0)
        stimulus_shape.setNVertices(shape_edges)
        # Run 'Begin Routine' code from trigger_tone
        pulse_started = False
        
        # create starting attributes for shape_resp
        shape_resp.keys = []
        shape_resp.rt = []
        _shape_resp_allKeys = []
        # store start times for trial
        trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial.tStart = globalClock.getTime(format='float')
        trial.status = STARTED
        thisExp.addData('trial.started', trial.tStart)
        trial.maxDuration = None
        # keep track of which components have finished
        trialComponents = trial.components
        for thisComponent in trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *stimulus_shape* updates
            
            # if stimulus_shape is starting this frame...
            if stimulus_shape.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                stimulus_shape.frameNStart = frameN  # exact frame index
                stimulus_shape.tStart = t  # local t and not account for scr refresh
                stimulus_shape.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stimulus_shape, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stimulus_shape.started')
                # update status
                stimulus_shape.status = STARTED
                stimulus_shape.setAutoDraw(True)
            
            # if stimulus_shape is active this frame...
            if stimulus_shape.status == STARTED:
                # update params
                pass
            
            # if stimulus_shape is stopping this frame...
            if stimulus_shape.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stimulus_shape.tStartRefresh + STIMULUS_DURATION-frameTolerance:
                    # keep track of stop time/frame for later
                    stimulus_shape.tStop = t  # not accounting for scr refresh
                    stimulus_shape.tStopRefresh = tThisFlipGlobal  # on global time
                    stimulus_shape.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stimulus_shape.stopped')
                    # update status
                    stimulus_shape.status = FINISHED
                    stimulus_shape.setAutoDraw(False)
            
            # *text_fixation_cross* updates
            
            # if text_fixation_cross is starting this frame...
            if text_fixation_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_fixation_cross.frameNStart = frameN  # exact frame index
                text_fixation_cross.tStart = t  # local t and not account for scr refresh
                text_fixation_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_fixation_cross, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_fixation_cross.status = STARTED
                text_fixation_cross.setAutoDraw(True)
            
            # if text_fixation_cross is active this frame...
            if text_fixation_cross.status == STARTED:
                # update params
                pass
            
            # if text_fixation_cross is stopping this frame...
            if text_fixation_cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_fixation_cross.tStartRefresh + iti-frameTolerance:
                    # keep track of stop time/frame for later
                    text_fixation_cross.tStop = t  # not accounting for scr refresh
                    text_fixation_cross.tStopRefresh = tThisFlipGlobal  # on global time
                    text_fixation_cross.frameNStop = frameN  # exact frame index
                    # update status
                    text_fixation_cross.status = FINISHED
                    text_fixation_cross.setAutoDraw(False)
            # Run 'Each Frame' code from trigger_tone
            if stimulus_shape.status == STARTED and not pulse_started:
                if shape_is_oddball:
                    dev.activate_line(bitmask=oddball_p300_code)
                elif shape_is_distractor:
                    dev.activate_line(bitmask=distractor_p300_code)
                else:
                    dev.activate_line(bitmask=regular_p300_code)
            
                pulse_started = True
            
            
            # *shape_resp* updates
            waitOnFlip = False
            
            # if shape_resp is starting this frame...
            if shape_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                shape_resp.frameNStart = frameN  # exact frame index
                shape_resp.tStart = t  # local t and not account for scr refresh
                shape_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(shape_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'shape_resp.started')
                # update status
                shape_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(shape_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(shape_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if shape_resp is stopping this frame...
            if shape_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > shape_resp.tStartRefresh + iti-frameTolerance:
                    # keep track of stop time/frame for later
                    shape_resp.tStop = t  # not accounting for scr refresh
                    shape_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    shape_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'shape_resp.stopped')
                    # update status
                    shape_resp.status = FINISHED
                    shape_resp.status = FINISHED
            if shape_resp.status == STARTED and not waitOnFlip:
                theseKeys = shape_resp.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=False)
                _shape_resp_allKeys.extend(theseKeys)
                if len(_shape_resp_allKeys):
                    shape_resp.keys = [key.name for key in _shape_resp_allKeys]  # storing all keys
                    shape_resp.rt = [key.rt for key in _shape_resp_allKeys]
                    shape_resp.duration = [key.duration for key in _shape_resp_allKeys]
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial
        trial.tStop = globalClock.getTime(format='float')
        trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial.stopped', trial.tStop)
        # check responses
        if shape_resp.keys in ['', [], None]:  # No response was made
            shape_resp.keys = None
        trials.addData('shape_resp.keys',shape_resp.keys)
        if shape_resp.keys != None:  # we had a response
            trials.addData('shape_resp.rt', shape_resp.rt)
            trials.addData('shape_resp.duration', shape_resp.duration)
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed N_TRIALS repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "__end__" ---
    # create an object to store info about Routine __end__
    __end__ = data.Routine(
        name='__end__',
        components=[text_thank_you, read_thank_you],
    )
    __end__.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    read_thank_you.setSound('resource/thank_you.wav', secs=2.7, hamming=True)
    read_thank_you.setVolume(1.0, log=False)
    read_thank_you.seek(0)
    # Run 'Begin Routine' code from trigger_trial_block_end
    # End of main experiment trial block
    dev.activate_line(bitmask=block_end_code)
    eyetracker.sendMessage(block_end_code)
    # no need to wait 500ms as this routine lasts 3.0s before experiment ends
    
    # store start times for __end__
    __end__.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    __end__.tStart = globalClock.getTime(format='float')
    __end__.status = STARTED
    __end__.maxDuration = None
    # keep track of which components have finished
    __end__Components = __end__.components
    for thisComponent in __end__.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "__end__" ---
    __end__.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_thank_you* updates
        
        # if text_thank_you is starting this frame...
        if text_thank_you.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_thank_you.frameNStart = frameN  # exact frame index
            text_thank_you.tStart = t  # local t and not account for scr refresh
            text_thank_you.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_thank_you, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_thank_you.status = STARTED
            text_thank_you.setAutoDraw(True)
        
        # if text_thank_you is active this frame...
        if text_thank_you.status == STARTED:
            # update params
            pass
        
        # if text_thank_you is stopping this frame...
        if text_thank_you.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_thank_you.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                text_thank_you.tStop = t  # not accounting for scr refresh
                text_thank_you.tStopRefresh = tThisFlipGlobal  # on global time
                text_thank_you.frameNStop = frameN  # exact frame index
                # update status
                text_thank_you.status = FINISHED
                text_thank_you.setAutoDraw(False)
        
        # *read_thank_you* updates
        
        # if read_thank_you is starting this frame...
        if read_thank_you.status == NOT_STARTED and t >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            read_thank_you.frameNStart = frameN  # exact frame index
            read_thank_you.tStart = t  # local t and not account for scr refresh
            read_thank_you.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_thank_you.status = STARTED
            read_thank_you.play()  # start the sound (it finishes automatically)
        
        # if read_thank_you is stopping this frame...
        if read_thank_you.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > read_thank_you.tStartRefresh + 2.7-frameTolerance or read_thank_you.isFinished:
                # keep track of stop time/frame for later
                read_thank_you.tStop = t  # not accounting for scr refresh
                read_thank_you.tStopRefresh = tThisFlipGlobal  # on global time
                read_thank_you.frameNStop = frameN  # exact frame index
                # update status
                read_thank_you.status = FINISHED
                read_thank_you.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_thank_you]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            __end__.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in __end__.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "__end__" ---
    for thisComponent in __end__.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for __end__
    __end__.tStop = globalClock.getTime(format='float')
    __end__.tStopRefresh = tThisFlipGlobal
    read_thank_you.pause()  # ensure sound has stopped at end of Routine
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if __end__.maxDurationReached:
        routineTimer.addTime(-__end__.maxDuration)
    elif __end__.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    # Run 'End Experiment' code from eeg
    # Stop EEG recording
    dev.activate_line(bitmask=127)  # trigger 127 will stop EEG
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
