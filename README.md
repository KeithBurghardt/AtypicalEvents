# AtypicalEvents

## Atypical Event Detection
Event prediction code includes feature selection tool (MRMR) [1], and event prediction for positive, negative, and atypical events. I also list the features selected for the hospital and aerospace datasets. Core of MRMR code is created by Shen Yan (shenyan@isi.edu). All other code is written by myself.

## Causal Analysis of Atypical Events: AtypicalEventCausalEffect.wls

We use Mathematica to infer the causal effect of atypical events. We namely explore the change in the absolute values of atypical events from the day before, the day of (treatment), and the day after an atypical event for each subject. We compare these results to the same days where a subject does not experience an atypical event that same day (control). The difference between control and treatment is the average treatement effect we estimate.

[1] Peng, H.C., Long, F., and Ding, C., "Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy," IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 27, No. 8, pp. 1226–1238, 2005.
