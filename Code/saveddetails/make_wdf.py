
import os
import rwpp as wv

path_data = os.getcwd() + r'\raw_wdata'
    
''' MINI-BATCH TRAINED DELTA-RULE '''
path_data += r'\minibatch'

# wdf_runs04   = wv.load(path_data + r'\wdf_runs_0-4.pkl')
# wdf_runs59   = wv.load(path_data + r'\wdf_runs_5-9.pkl')
# wdf_ep0      = wv.load(path_data + r'\wdf_0ep.pkl')
# wdf_ep40     = wv.load(path_data + r'\wdf_40ep.pkl')
# wdf_run10    = wv.load(path_data + r'\wdf_runs_10.pkl')
# wdf_runs1113 = wv.load(path_data + r'\wdf_runs_11-13.pkl')
# wdf_runs1416 = wv.load(path_data + r'\wdf_runs_14-16.pkl')
# wdf_runs1719 = wv.load(path_data + r'\wdf_runs_17-19.pkl')

# wdf = wv._join([wdf_runs04, wdf_runs59])
# for i in range(list(wdf.keys()).__len__()):
#     wdf[i] = wv.insertion(wdf[i], wdf_ep0[i], 0)
#     wdf[i] = wv.insertion(wdf[i], wdf_ep40[i], 40)
#     wdf[i] = wdf[i].sort_index()
# #end
# wdf = wv._join([ wdf, wdf_run10, wdf_runs1113, wdf_runs1416, wdf_runs1719 ])


wdf_runs04 = wv.load(path_data + r'\sz_discrimination_psydata_df_runs_0-4.pkl')
wdf_runs59 = wv.load(path_data + r'\sz_discrimination_psydata_df_runs_5-9.pkl')
wdf_runs1014 = wv.load(path_data + r'\sz_discrimination_psydata_df_runs_10-14.pkl')
wdf_runs1519 = wv.load(path_data + r'\sz_discrimination_psydata_df_runs_15-19.pkl')


wdf = wv._join( [wdf_runs04, wdf_runs59, wdf_runs1014, wdf_runs1519] )

wdf = wv.convert_to_numeric(wdf)
wdf = wv.outlier_remove(wdf, threshold = 3)
wdf = wv.clean(wdf, scale_stds = 2)

wv.save(path_data + r'\..\wdf_preprocss.pkl', wdf)