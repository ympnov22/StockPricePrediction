import pandas as pd
import numpy as np

pd_traing_loss = pd.read_csv( 'H1010/training_loss.csv', header = None, names = ('H'))
pd_traing_loss['H1030'] = pd.read_csv( 'H1030/training_loss.csv',header = None)
pd_traing_loss['H1050'] = pd.read_csv( 'H1050/training_loss.csv',header = None)
pd_traing_loss['H1080'] = pd.read_csv( 'H1080/training_loss.csv',header = None)
pd_traing_loss['H3010'] = pd.read_csv( 'H3010/training_loss.csv',header = None)
pd_traing_loss['H3030'] = pd.read_csv( 'H3030/training_loss.csv',header = None)
pd_traing_loss['H3050'] = pd.read_csv( 'H3050/training_loss.csv',header = None)
pd_traing_loss['H3080'] = pd.read_csv( 'H3080/training_loss.csv',header = None)
pd_traing_loss['H5010'] = pd.read_csv( 'H5010/training_loss.csv',header = None)
pd_traing_loss['H5030'] = pd.read_csv( 'H5030/training_loss.csv',header = None)
pd_traing_loss['H5050'] = pd.read_csv( 'H5050/training_loss.csv',header = None)
pd_traing_loss['H5080'] = pd.read_csv( 'H5080/training_loss.csv',header = None)
pd_traing_loss['H8010'] = pd.read_csv( 'H8010/training_loss.csv',header = None)
pd_traing_loss['H8030'] = pd.read_csv( 'H8030/training_loss.csv',header = None)
pd_traing_loss['H8050'] = pd.read_csv( 'H8050/training_loss.csv',header = None)
pd_traing_loss['H8080'] = pd.read_csv( 'H8080/training_loss.csv',header = None)
#print(pd_traing_loss)
np.savetxt("training_loss_marged.csv", pd_traing_loss.values , delimiter=",")

pd_testing_loss = pd.read_csv( 'H1010/testing_loss.csv', header = None, names = ('H'))
pd_testing_loss['H1030'] = pd.read_csv( 'H1030/testing_loss.csv',header = None)
pd_testing_loss['H1050'] = pd.read_csv( 'H1050/testing_loss.csv',header = None)
pd_testing_loss['H1080'] = pd.read_csv( 'H1080/testing_loss.csv',header = None)
pd_testing_loss['H3010'] = pd.read_csv( 'H3010/testing_loss.csv',header = None)
pd_testing_loss['H3030'] = pd.read_csv( 'H3030/testing_loss.csv',header = None)
pd_testing_loss['H3050'] = pd.read_csv( 'H3050/testing_loss.csv',header = None)
pd_testing_loss['H3080'] = pd.read_csv( 'H3080/testing_loss.csv',header = None)
pd_testing_loss['H5010'] = pd.read_csv( 'H5010/testing_loss.csv',header = None)
pd_testing_loss['H5030'] = pd.read_csv( 'H5030/testing_loss.csv',header = None)
pd_testing_loss['H5050'] = pd.read_csv( 'H5050/testing_loss.csv',header = None)
pd_testing_loss['H5080'] = pd.read_csv( 'H5080/testing_loss.csv',header = None)
pd_testing_loss['H8010'] = pd.read_csv( 'H8010/testing_loss.csv',header = None)
pd_testing_loss['H8030'] = pd.read_csv( 'H8030/testing_loss.csv',header = None)
pd_testing_loss['H8050'] = pd.read_csv( 'H8050/testing_loss.csv',header = None)
pd_testing_loss['H8080'] = pd.read_csv( 'H8080/testing_loss.csv',header = None)
#print(pd_traing_loss)
np.savetxt("testing_loss_marged.csv", pd_testing_loss.values , delimiter=",")

pd_accuracy = pd.read_csv( 'H1010/acuracy.csv', header = None, names = ('H'))
pd_accuracy['H1030'] = pd.read_csv( 'H1030/acuracy.csv',header = None)
pd_accuracy['H1050'] = pd.read_csv( 'H1050/acuracy.csv',header = None)
pd_accuracy['H1080'] = pd.read_csv( 'H1080/acuracy.csv',header = None)
pd_accuracy['H3010'] = pd.read_csv( 'H3010/acuracy.csv',header = None)
pd_accuracy['H3030'] = pd.read_csv( 'H3030/acuracy.csv',header = None)
pd_accuracy['H3050'] = pd.read_csv( 'H3050/acuracy.csv',header = None)
pd_accuracy['H3080'] = pd.read_csv( 'H3080/acuracy.csv',header = None)
pd_accuracy['H5010'] = pd.read_csv( 'H5010/acuracy.csv',header = None)
pd_accuracy['H5030'] = pd.read_csv( 'H5030/acuracy.csv',header = None)
pd_accuracy['H5050'] = pd.read_csv( 'H5050/acuracy.csv',header = None)
pd_accuracy['H5080'] = pd.read_csv( 'H5080/acuracy.csv',header = None)
pd_accuracy['H8010'] = pd.read_csv( 'H8010/acuracy.csv',header = None)
pd_accuracy['H8030'] = pd.read_csv( 'H8030/acuracy.csv',header = None)
pd_accuracy['H8050'] = pd.read_csv( 'H8050/acuracy.csv',header = None)
pd_accuracy['H8080'] = pd.read_csv( 'H8080/acuracy.csv',header = None)
#print(pd_traing_loss)
np.savetxt("accuracy_marged.csv", pd_accuracy.values , delimiter=",")

pd_profit = pd.read_csv( 'H1010/profit.csv', header = None, names = ('H'))
pd_profit['H1030'] = pd.read_csv( 'H1030/profit.csv',header = None)
pd_profit['H1050'] = pd.read_csv( 'H1050/profit.csv',header = None)
pd_profit['H1080'] = pd.read_csv( 'H1080/profit.csv',header = None)
pd_profit['H3010'] = pd.read_csv( 'H3010/profit.csv',header = None)
pd_profit['H3030'] = pd.read_csv( 'H3030/profit.csv',header = None)
pd_profit['H3050'] = pd.read_csv( 'H3050/profit.csv',header = None)
pd_profit['H3080'] = pd.read_csv( 'H3080/profit.csv',header = None)
pd_profit['H5010'] = pd.read_csv( 'H5010/profit.csv',header = None)
pd_profit['H5030'] = pd.read_csv( 'H5030/profit.csv',header = None)
pd_profit['H5050'] = pd.read_csv( 'H5050/profit.csv',header = None)
pd_profit['H5080'] = pd.read_csv( 'H5080/profit.csv',header = None)
pd_profit['H8010'] = pd.read_csv( 'H8010/profit.csv',header = None)
pd_profit['H8030'] = pd.read_csv( 'H8030/profit.csv',header = None)
pd_profit['H8050'] = pd.read_csv( 'H8050/profit.csv',header = None)
pd_profit['H8080'] = pd.read_csv( 'H8080/profit.csv',header = None)
#print(pd_traing_loss)
np.savetxt("profit_marged.csv", pd_profit.values , delimiter=",")