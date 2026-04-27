import pandas as pd
from pm4py.objects.log.obj import EventLog,Event,Trace

"""
Create a single trace, given some group id, key values from dataframe groupby, trace instance number and some dataframe of events to handle for this trace. The optional trace_attrs parameter determines attributes to promote from event to trace level.
"""
def convert_trace(trace_id:str, event_col:str, df:pd.DataFrame,
                  trace_attrs=None) -> Trace:
    # build a trace object
    trace = Trace()
    # to add attributes to a trace, use the .attribute member of the trace
    # .attribtues is a dictionary
    trace.attributes['concept:name'] = trace_id
    if trace_attrs:
        for attr in trace_attrs:
            trace.attributes[attr] = trace_attrs[attr][trace_id]
    # convert rows into events
    #df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    #df = df.sort_values('time:timestamp')
    for _,event_data in df.iterrows():
        event = Event()
        edd = event_data.to_dict()
        event['concept:name'] = edd[event_col]
        for key in edd:
            event[key] = event_data[key]
        trace.append(event)
    return trace

"""
Convert dataframe to XES log
"""
def convert_to_log(df,casecol,eventcol,trace_attrs=None,
                   logname='Event log'):
    event_log = EventLog(
            **{
                "attributes" : {
                    "concept:name" : logname
                }
            } )
    traces = []
    for dgi,tdf in df.groupby(casecol):
        traces.append(convert_trace(dgi,eventcol,tdf,trace_attrs=trace_attrs))
    event_log._list = traces
    return event_log