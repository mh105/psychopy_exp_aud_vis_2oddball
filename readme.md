# Auditory and visual oddball tasks (no pupil)
Last edit: 03/17/2026

## Edit history
- 03/17/2026 by Jenny Tou and Alex He - initial draft of separate auditory and visual double oddball tasks without pillometry

## Description
This experiment includes both auditory and visual oddball tasks designed to elicit the classical P300 event-related potential (ERP) response to oddballs in a target detection paradigm. A number of variations of the task design have been used in the literature as summarized by:

Polich, J. (2007). Updating P300: an integrative theory of P3a and P3b. Clinical neurophysiology, 118(10), 2128-2148.

Double oddballs are used in order to observe separate P3a and P3b components of the P300 response. One oddball is the oddball target - eliciting P3b, while the other oddball is the oddball distractor, eliciting P3a.

We have implemented both auditory and visual modalities. For the auditory task, subjects listen to tones with eyes closed to reduce the chances of eye blink and movement artifacts in the EEG recordings. For the visual task, subjects view shapes on screen. A total of 200 trials are administered in each modality during a session.

Past literature has disagreed on whether behavioral responses should be elicited for both targets and non-targets, as well as on the percentage of oddballs. We have decided to use 20% oddballs (10% target, 10% distractor) to increase the rarity and strength of the P300 response; for the same reason, subjects are only responding to the target (oddball) tones/shapes and not producing any behavioral responses for regular (common) or distractor (oddball) stimuli.

## Outcome measures
- P300 ERP waveform derived from contrasting between target/distractor and regular trial in both auditory and visual modalities
