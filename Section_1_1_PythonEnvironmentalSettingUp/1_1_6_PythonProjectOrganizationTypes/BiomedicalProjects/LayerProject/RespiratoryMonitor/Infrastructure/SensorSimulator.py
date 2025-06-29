#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
import random;

#*********************************************************************************************************#
#********************************************* SUPPORTING FUNCTIONS **************************************#
#*********************************************************************************************************#
def simulateBreathingSignal(durationInSec=60, averageBPM=16):
    '''
    Simulates a sequence of breathing event timestamps over a given duration.

    This function generates a list of timestamps representing synthetic breathing 
    events, spaced roughly according to a target average breaths per minute (BPM).
    The intervals are randomized within �20% of the average to mimic natural variability.

    Args:
        durationInSec (int, optional): Total simulation duration in seconds. Defaults to 60.
        averageBPM (float, optional): Average number of breaths per minute. Defaults to 16.

    Returns:
        list of float: Simulated timestamps (in seconds) for each detected breath, 
        rounded to 2 decimal places.
    '''

    interval = 60 / averageBPM;
    timeStamps = [];
    t = 0;
    while t < durationInSec:
        t += random.uniform(interval * 0.8, interval * 1.2);
        timeStamps.append(round(t, 2));
    return timeStamps;
