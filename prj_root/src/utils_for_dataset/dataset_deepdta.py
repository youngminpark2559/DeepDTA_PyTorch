import json
import csv
import pickle
import collections
from collections import OrderedDict
import numpy as np
import pandas as pd
from random import shuffle
from copy import deepcopy

# ================================================================================
import torch.utils.data as data

# ================================================================================
unique_characters_for_protein={
  "A":1,"C":2,"B":3,"E":4,"D":5,"G":6,"F":7,"I":8,"H":9,"K":10,"M":11,"L":12,"O":13,"N":14,
  "Q":15,"P":16,"S":17,"R":18,"U":19,"T":20,"W":21,"V":22,"Y":23,"X":24,"Z":25}

length_of_unique_characters_for_protein=len(unique_characters_for_protein)
# print("length_of_unique_characters_for_protein",length_of_unique_characters_for_protein)
# length_of_unique_characters_for_protein 25

# ================================================================================
unique_characters_for_can_smiles={
  "#":1,"%":2,")":3,"(":4,"+":5,"-":6,".":7,"1":8,"0":9,"3":10,"2":11,"5":12,"4":13,"7":14,"6":15,"9":16,"8":17,"=":18,
  "A":19,"C":20,"B":21,"E":22,"D":23,"G":24,"F":25,"I":26,"H":27,"K":28,"M":29,"L":30,"O":31,"N":32,"P":33,"S":34,
  "R":35,"U":36,"T":37,"W":38,"V":39,"Y":40,"[":41,"Z":42,"]":43,"_":44,"a":45,"c":46,"b":47,"e":48,"d":49,"g":50,
  "f":51,"i":52,"h":53,"m":54,"l":55,"o":56,"n":57,"s":58,"r":59,"u":60,"t":61,"y":62}

length_of_unique_characters_for_can_smiles=len(unique_characters_for_can_smiles)
# print("length_of_unique_characters_for_can_smiles",length_of_unique_characters_for_can_smiles)
# length_of_unique_characters_for_can_smiles 62

# ================================================================================
unique_characters_for_iso_smiles={
  "#":29,"%":30,")":31,"(":1,"+":32,"-":33,"/":34,".":2,"1":35,"0":3,"3":36,"2":4,"5":37,"4":5,"7":38,"6":6,"9":39,
  "8":7,"=":40,"A":41,"@":8,"C":42,"B":9,"E":43,"D":10,"G":44,"F":11,"I":45,"H":12,"K":46,"M":47,"L":13,"O":48,"N":14,
  "P":15,"S":49,"R":16,"U":50,"T":17,"W":51,"V":18,"Y":52,"[":53,"Z":19,"]":54,"\\":20,"a":55,"c":56,"b":21,"e":57,
  "d":22,"g":58,"f":23,"i":59,"h":24,"m":60,"l":25,"o":61,"n":26,"s":62,"r":27,"u":63,"t":28,"y":64}

length_of_unique_characters_for_iso_smiles=len(unique_characters_for_iso_smiles)
# print("length_of_unique_characters_for_iso_smiles",length_of_unique_characters_for_iso_smiles)
# length_of_unique_characters_for_iso_smiles 64

# ================================================================================
def label_smiles(line,MAX_SMI_LEN,smi_ch_ind):
  # print("line",line)
  # COC1=C(C=C2C(=C1)CCN=C2C3=CC(=C(C=C3)Cl)Cl)Cl

  # print("MAX_SMI_LEN",MAX_SMI_LEN)
  # MAX_SMI_LEN 100

  X = np.zeros(MAX_SMI_LEN)
  # print("X",X)
  # print("X",X.shape)
  # X [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  #    0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  #    0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  #    0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  #    0. 0. 0. 0.]
  # X (100,)

  for i,ch in enumerate(line[:MAX_SMI_LEN]):
    X[i]=smi_ch_ind[ch]

  return X

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
  X = np.zeros(MAX_SEQ_LEN)

  for i, ch in enumerate(line[:MAX_SEQ_LEN]):
    X[i] = smi_ch_ind[ch]

  return X

def nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds, args):

    bestparamlist = []
    test_set, outer_train_sets = read_sets(args)

    # ================================================================================
    foldinds = len(outer_train_sets)
    # print("foldinds",foldinds)
    # 5

    # ================================================================================
    # @ Create indices array for dataset 
    test_sets = []
    val_sets = []
    train_sets = []

    # logger.info('Start training')
    for val_foldind in range(foldinds):
        # ================================================================================
        # @ Validation fold

        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)

        # ================================================================================
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]

        # ================================================================================
        # @ Train fold

        train_sets.append(otherfoldsinds)

        # ================================================================================
        # @ Test fold

        test_sets.append(test_set)

        # ================================================================================
        # print("val set", str(len(val_fold)))
        # print("train set", str(len(otherfoldsinds)))

    # print("test_sets",test_sets)
    # print("val_sets",val_sets)
    # print("train_sets",train_sets)

    # print("test_sets",np.array(test_sets).shape)
    # print("val_sets",np.array(val_sets).shape)
    # print("train_sets",np.array(train_sets).shape)
    # test_sets (5, 19709)
    # val_sets (5, 19709)
    # train_sets (5, 78836)

    # ================================================================================
    # @ Perform training

    bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = general_nfold_cv(
      XD, XT,  Y, label_row_inds, label_col_inds, measure, runmethod, FLAGS, train_sets, val_sets)
    # afaf 4: general_nfold_cv(XD, XT,  Y, label_row_inds, label_col_inds, measure, runmethod, FLAGS, train_sets, val_sets)
    
    # ================================================================================
    #print("Test Set len", str(len(test_set)))
    #print("Outer Train Set len", str(len(outer_train_sets)))
    bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(XD, XT,  Y, label_row_inds, label_col_inds, 
                                                                                                measure, runmethod, FLAGS, train_sets, test_sets)
    
    testperf = all_predictions[bestparamind]##pointer pos 

    logging("---FINAL RESULTS-----", FLAGS)
    logging("best param index = %s,  best param = %.5f" % 
            (bestparamind, bestparam), FLAGS)


    testperfs = []
    testloss= []

    avgperf = 0.

    for test_foldind in range(len(test_sets)):
        foldperf = all_predictions[bestparamind][test_foldind]
        foldloss = all_losses[bestparamind][test_foldind]
        testperfs.append(foldperf)
        testloss.append(foldloss)
        avgperf += foldperf

    avgperf = avgperf / len(test_sets)
    avgloss = np.mean(testloss)
    teststd = np.std(testperfs)

    logging("Test Performance CI", FLAGS)
    logging(testperfs, FLAGS)
    logging("Test Performance MSE", FLAGS)
    logging(testloss, FLAGS)

    return avgperf, avgloss, teststd

def general_nfold_cv(XD, XT,  Y, label_row_inds, label_col_inds, prfmeasure, runmethod, FLAGS, labeled_sets, val_sets): ## BURAYA DA FLAGS LAZIM????
    
    paramset1 = FLAGS.num_windows                                      #[32]#[32,  512] #[32, 128]  # filter numbers
    paramset2 = FLAGS.smi_window_lengths                               #[4, 8]#[4,  32] #[4,  8] #filter length smi
    paramset3 = FLAGS.seq_window_lengths                               #[8, 12]#[64,  256] #[64, 192]#[8, 192, 384]
    # epoch = FLAGS.num_epoch
    epoch = 2
    batchsz = FLAGS.batch_size
    # print("paramset1",paramset1)
    # print("paramset2",paramset2)
    # print("paramset3",paramset3)
    # print("epoch",epoch)
    # print("batchsz",batchsz)
    # paramset1 [32]
    # paramset2 [4, 8]
    # paramset3 [8, 12]
    # epoch 100
    # batchsz 256

    # ================================================================================
    logging("---Parameter Search-----", FLAGS)

    # ================================================================================
    w = len(val_sets)
    h = len(paramset1) * len(paramset2) * len(paramset3)
    # print("w",w)
    # print("h",h)
    # w 5
    # h 4

    # ================================================================================
    all_predictions = [[0 for x in range(w)] for y in range(h)] 
    all_losses = [[0 for x in range(w)] for y in range(h)] 
    print(all_predictions)

    # ================================================================================
    # @ Perform validation

    for foldind in range(len(val_sets)):
        # c valinds: validation smiles data indices
        valinds = val_sets[foldind]
        # c labeledinds: validation protein data indices
        labeledinds = labeled_sets[foldind]
        # c Y_train: validation label data
        Y_train = np.mat(np.copy(Y))
        # print("Y_train",Y_train)
        # print("Y_train",Y_train.shape)
        # [[11.1                nan         nan ...         nan         nan       nan]
        #  [11.1                nan         nan ...         nan         nan       nan]
        #  [12.1        11.99999842         nan ...         nan 14.40016227 11.30000024]
        #  ...
        #  [        nan         nan         nan ...         nan         nan  12.637602]
        #  [        nan         nan         nan ...         nan         nan       nan]
        #  [        nan         nan         nan ...         nan         nan       nan]]
        # Y_train (2111, 229)

        # ================================================================================
        params = {}
        XD_train = XD
        XT_train = XT
        # print("XD_train",XD_train.shape)
        # print("XT_train",XT_train.shape)
        # XD_train (2111, 100)
        # XT_train (229, 1000)

        # ================================================================================
        # @ Prepare data (train)

        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]

        XD_train = XD[trrows]
        XT_train = XT[trcols]

        train_drugs, train_prots, train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)
        # print("train_drugs",train_drugs)
        # print("train_prots",train_prots)
        # print("train_Y",train_Y)
        # print("train_drugs",train_drugs.shape)
        # print("train_prots",train_prots.shape)
        # print("train_Y",np.array(train_Y).shape)
        # train_drugs (78836, 100)
        # train_prots (78836, 1000)
        # train_Y (78836,)


        # ================================================================================
        # @ Prepare data (validation)

        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]
        #print("terows", str(terows), str(len(terows)))
        #print("tecols", str(tecols), str(len(tecols)))

        val_drugs, val_prots, val_Y = prepare_interaction_pairs(XD, XT, Y, terows, tecols)

        # ================================================================================
        pointer = 0
       
        for param1ind in range(len(paramset1)): # hidden neurons
            param1value = paramset1[param1ind]
            # print("param1value",param1value)
            # param1value 32

            for param2ind in range(len(paramset2)): # learning rate
                param2value = paramset2[param2ind]
                # print("param2value",param2value)
                # param2value 4

                for param3ind in range(len(paramset3)):
                    param3value = paramset3[param3ind]
                    # print("param3value",param3value)
                    # param3value 8

                    # ================================================================================
                    # runmethod=build_combined_categorical
                    gridmodel = runmethod(FLAGS, param1value, param2value, param3value)

                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

                    # ================================================================================
                    # @ Prepare dataset

                    train_drugs_data=np.array(train_drugs)
                    train_proteins_data=np.array(train_prots)
                    train_affinity_label_data=np.array(train_Y)
                    # print("train_drugs_data",train_drugs_data.shape)
                    # print("train_proteins_data",train_proteins_data.shape)
                    # print("train_affinity_label_data",train_affinity_label_data.shape)
                    # train_drugs_data (78836, 100)
                    # train_proteins_data (78836, 1000)
                    # train_affinity_label_data (78836,)

                    validation_drugs_data=np.array(val_drugs)
                    validation_proteins_data=np.array(val_prots)
                    validation_affinity_label_data=np.array(val_Y)
                    # print("validation_drugs_data",validation_drugs_data.shape)
                    # print("validation_proteins_data",validation_proteins_data.shape)
                    # print("validation_affinity_label_data",validation_affinity_label_data.shape)
                    # validation_drugs_data (19709, 100)
                    # validation_proteins_data (19709, 1000)
                    # validation_affinity_label_data (19709,)

                    validation_drugs_data=validation_drugs_data[:255,:]
                    validation_proteins_data=validation_proteins_data[:255,:]
                    validation_affinity_label_data=validation_affinity_label_data[:255]
                    
                    # ================================================================================
                    # @ Perform training

                    gridres = gridmodel.fit(
                        ([train_drugs_data,train_proteins_data]),train_affinity_label_data, 
                        batch_size=batchsz,epochs=epoch, 
                        validation_data=(([validation_drugs_data,validation_proteins_data]),validation_affinity_label_data),
                        shuffle=False,callbacks=[es])
                    
                    # ================================================================================
                    # @ Perform validation

                    predicted_labels = gridmodel.predict([validation_drugs_data,validation_proteins_data])

                    # ================================================================================
                    # @ Evaluate validation predictions

                    loss, rperf2 = gridmodel.evaluate(([validation_drugs_data,validation_proteins_data]),validation_affinity_label_data,verbose=0)
                    rperf = prfmeasure(val_Y, predicted_labels)
                    rperf = rperf[0]
                    
                    # ================================================================================
                    logging("P1 = %d,  P2 = %d, P3 = %d, Fold = %d, CI-i = %f, CI-ii = %f, MSE = %f" % (param1ind, param2ind, param3ind, foldind, rperf, rperf2, loss), FLAGS)

                    # ================================================================================
                    plotLoss(gridres, param1ind, param2ind, param3ind, foldind)

                    # ================================================================================
                    all_predictions[pointer][foldind] =rperf #TODO FOR EACH VAL SET allpredictions[pointer][foldind]
                    all_losses[pointer][foldind]= loss

                    pointer +=1

    bestperf = -float('Inf')
    bestpointer = None

    # ================================================================================
    best_param_list = []
    
    # ================================================================================
    # @ Take average according to folds, then chooose best params

    pointer = 0
    for param1ind in range(len(paramset1)):
            for param2ind in range(len(paramset2)):
                for param3ind in range(len(paramset3)):
                
                    avgperf = 0.
                    for foldind in range(len(val_sets)):
                        foldperf = all_predictions[pointer][foldind]
                        avgperf += foldperf
                    avgperf /= len(val_sets)
                    
                    if avgperf > bestperf:
                        bestperf = avgperf
                        bestpointer = pointer
                        best_param_list = [param1ind, param2ind, param3ind]

                    pointer=pointer+1
    
    print("pointer",pointer)
    print("bestpointer",bestpointer)
    print("best_param_list",best_param_list)
    print("all_predictions",all_predictions)
    print("all_losses",all_losses)
    
    return bestpointer, best_param_list, bestperf, all_predictions, all_losses

def read_sets(args): ### fpath should be the dataset folder /kiba/ or /davis/
    fpath=args.dataset_path
    setting_no=args.problem_type
    # print("fpath",fpath)
    # print("setting_no",setting_no)
    # data/kiba/
    # 1

    # ================================================================================
    test_fold = json.load(open(fpath + "/folds/test_fold_setting" + str(setting_no)+".txt"))
    train_folds = json.load(open(fpath + "/folds/train_fold_setting" + str(setting_no)+".txt"))
    # print("test_fold",test_fold)
    # print("train_folds",train_folds)
    # [34121, 51548, 12611, 104850, 23744, 79716, 47565, 6166, 113707, 26852, 29930, 90721, 58109, 6584, 
    # [[113703, 51222, 98078, 29622, 80976, 112045, 13772, 30711, 60861, 37749, 81354, 63562, 95894, 4304, 66134, 113956, 101753, 52209, 77547, 4971, 97974

    # print("test_fold",np.array(test_fold).shape)
    # print("train_folds",np.array(train_folds).shape)
    # (19709,)
    # (5, 19709)
    
    return test_fold, train_folds

# ================================================================================
class DeepDTA_Dataset(data.Dataset):
  def __init__(self,args):

    fpath = args.dataset_path
    # print("fpath",fpath)
    # ../Data/kiba

    # ================================================================================
    smile_data=json.load(open(fpath+"/ligands_can.txt"),object_pairs_hook=OrderedDict)
    proteins_data=json.load(open(fpath+"/proteins.txt"),object_pairs_hook=OrderedDict)
    smile_protein_binding_affinity_score_data=pickle.load(open(fpath+"/Y","rb"),encoding='latin1')
    # print("smile_data",smile_data)
    # print("proteins_data",proteins_data)
    # print("smile_protein_binding_affinity_score_data",smile_protein_binding_affinity_score_data)
    # OrderedDict([('CHEMBL1087421', 'COC1=C(C=C2C(=C1)CCN=C2C3=CC(=C(C=C3)Cl)Cl)Cl'), ('CHEMBL1088633', 'COC1=C(C=C2C(=C1)CCN=C2C3=CC(=CC=C3)Cl)Cl'),
    # OrderedDict([('O00141', 'MTVKTEAAKGTLTYSRMRGMVAILIAFMKQRRMGLNDFIQKIANNSYACKHPEVQSILKISQPQEPELMNANPSPPPSPSQQINLGPSSNPHAKPSDFHFLKVIGKGSFGKVLLARHKAEEVFYAVKVLQKKAILKKKEEKHIMSERNVLLKNVKHPFLVGLHFSFQTADKLYFVLDYINGGELFYHLQRERCFLEPRARFYAAEIASALGYLHSLNIVYRDLKPENILLDSQGHIVLTDFGLCKENIEHNSTTSTFCGTPEYLAPEVLHKQPYDRTVDWWCLGAVLYEMLYGLPPFYSRNTAEMYDNILNKPLQLKPNITNSARHLLEGLLQKDRTKRLGAKDDFMEIKSHVFFSLINWDDLINKKITPPFNPNVSGPNDLRHFDPEFTEEPVPNSIGKSPDSVLVTASVKEAAEAFLGFSYAPPTDSFL'), ('O00311', 'MEASLGIQMDEPMAFSPQRDRFQAEGSLKKNEQNFKLAGVKKDIEKLYEAVPQLSNVFKIEDKIGEGTFSSVYLATAQLQVGPEEKIALKHLIPTSHPIRIAAELQCLTVAGGQDNVMGVKYCFRKNDHVVIAMPYLEHESFLDILNSLSFQEVREYMLNLFKALKRIHQFGIVHRDVKPSNFLYNRRLKKYALVDFGLAQGTHDTKIELLKFVQSEAQQERCSQNKSHIITGNKIPLSGPVPKELDQQSTTKASVKRPYTNAQIQIKQGKDGKEGSVGLSVQRSVFGERNFNIHSSISHESPAVKLMKQSKTVDVLSRKLATKKKAISTKVMNSAVMRKTASSCPASLTCDCYATDKVCSICLSRRQQVAPRAGTPGFRAPEVLTKCPNQTTAIDMWSAGVIFLSLLSGRYPFYKASDDLTALAQIMTIRGSRETIQAAKTFGKSILCSKEVPAQDLRKLCERLRGMDSSTPKLTSDIQGHASHQPAISEKTDHKASCLVQTPPGQYSGNSFKKGDSNSCEHCFDEYNTNLEGWNEVPDEAYDLLDKLLDLNPASRITAEEALLHPFFKDMSL'), ('O00329', 'MPPGVDCPMEFWTKEENQSVVVDFLLPTGVYLNFPVSRNANLSTIKQLLWHRAQYEPLFHMLSGPEAYVFTCINQTAEQQELEDEQRRLCDVQPFLPVLRLVAREGDRVKKLINSQISLLIGKGLHEFDSLCDPEVNDFRAKMCQFCEEAAARRQQLGWEAWLQYSFPLQLEPSAQTWGPGTLRLPNRALLVNVKFEGSEESFTFQVSTKDVPLALMACALRKKATVFRQPLVEQPEDYTLQVNGRHEYLYGSYPLCQFQYICSCLHSGLTPHLTMVHSSSILAMRDEQSNPAPQVQKPRAKPPPIPAKKPSSVSLWSLEQPFRIELIQGSKVNADERMKLVVQAGLFHGNEMLCKTVSSSEVSVCSEPVWKQRLEFDINICDLPRMARLCFALYAVIEKAKKARSTKKKSKKADCPIAWANLMLFDYKDQLKTGERCLYMWPSVPDEKGELLNPTGTVRSNPNTDSAAALLICLPEVAPHPVYYPALEKILELGRHSECVHVTEEEQLQLREILERRGSGELYEHEKDLVWKLRHEVQEHFPEALARLLLVTKWNKHEDVAQMLYLLCSWPELPVLSALELLDFSFPDCHVGSFAIKSLRKLTDDELFQYLLQLVQVLKYESYLDCELTKFLLDRALANRKIGHFLFWHLRSEMHVPSVALRFGLILEAYCRGSTHHMKVLMKQGEALSKLKALNDFVKLSSQKTPKPQTKELMHLCMRQEAYLEALSHLQSPLDPSTLLAEVCVEQCTFMDSKMKPLWIMYSNEEAGSGGSVGIIFKNGDDLRQDMLTLQMIQLMDVLWKQEGLDLRMTPYGCLPTGDRTGLIEVVLRSDTIANIQLNKSNMAATAAFNKDALLNWLKSKNPGEALDRAIEEFTLSCAGYCVATYVLGIGDRHSDNIMIRESGQLFHIDFGHFLGNFKTKFGINRERVPFILTYDFVHVIQQGKTNNSEKFERFRGYCERAYTILRRHGLLFLHLFALMRAAGLPELSCSKDIQYLKDSLALGKTEEEALKHFRVKFNEALRESWKTKVNWLAHNVSKDNRQ'),
    # [[11.1                nan         nan ...         nan         nan          nan]
    #  [11.1                nan         nan ...         nan         nan          nan]
    #  [12.1        11.99999842         nan ...         nan 14.40016227  11.30000024]
    #  ...
    #  [        nan         nan         nan ...         nan         nan  12.637602  ]
    #  [        nan         nan         nan ...         nan         nan          nan]
    #  [        nan         nan         nan ...         nan         nan          nan]]

    # ================================================================================
    # if args.is_log:
    #     ten_power_9=math.pow(10,9)
    #     # print("ten_power_9",ten_power_9)
    #     Y = -(np.log10(Y/(math.pow(10,9))))

    # ================================================================================
    # print("with_label",with_label)
    # True

    with_label=True

    int_based_label_of_all_smiles_sequences=[]
    int_based_label_of_all_protein_sequences=[]

    if with_label:
        for d in smile_data.keys():
            # print("d",d)
            # CHEMBL1087421
            
            # ================================================================================
            one_drug_in_string=smile_data[d]
            # print("one_drug_in_string",one_drug_in_string)
            # drug_structure_in_string COC1=C(C=C2C(=C1)CCN=C2C3=CC(=C(C=C3)Cl)Cl)Cl

            # ================================================================================
            int_based_label_of_smiles_sequence=label_smiles(
              one_drug_in_string,int(args.max_length_of_smiles_sequence),unique_characters_for_iso_smiles)
            # print("int_based_label_of_smiles_sequence",int_based_label_of_smiles_sequence)
            # [42. 48. 42. 35. 40. 42.  1. 42. 40. 42.  4. 42.  1. 40. 42. 35. 31. 42.
            #  42. 14. 40. 42.  4. 42. 36. 40. 42. 42.  1. 40. 42.  1. 42. 40. 42. 36.
            #  31. 42. 25. 31. 42. 25. 31. 42. 25.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]

            int_based_label_of_all_smiles_sequences.append(int_based_label_of_smiles_sequence)

        for t in proteins_data.keys():
            # print("self.SEQLEN",self.SEQLEN)
            # print("self.charseqset",self.charseqset)
            # self.SEQLEN 1000
            # self.charseqset {'A': 1, 'C': 2, 'B': 3, 'E': 4, 'D': 5, 'G': 6, 'F': 7, 'I': 8, 'H': 9, 'K': 10, 'M': 11, 'L': 12, 'O': 13, 'N': 14, 'Q': 15, 'P': 16, 'S': 17, 'R': 18, 'U': 19, 'T': 20, 'W': 21, 'V': 22, 'Y': 23, 'X': 24, 'Z': 25}

            # print("proteins[t]",proteins[t])
            # afaf

            int_based_label_of_protein_sequence=label_sequence(
              proteins_data[t],int(args.max_length_of_protein_sequence),unique_characters_for_protein)
            int_based_label_of_all_protein_sequences.append(int_based_label_of_protein_sequence)

            # print("int_based_label_of_protein_sequence",int_based_label_of_protein_sequence)
            # print("int_based_label_of_protein_sequence",int_based_label_of_protein_sequence)
            # [11. 20. 22. 10. 20.  4.  1.  1. 10.  6. 20. 12. 20. 23. 17. 18. 11. 18.
            #   6. 11. 22.  1.  8. 12.  8.  1.  7. 11. 10. 15. 18. 18. 11.  6. 12. 14.
            #   5.  7.  8. 15. 10.  8.  1. 14. 14. 17. 23.  1.  2. 10.  9. 16.  4. 22.
            # 15. 17.  8. 12. 10.  8. 17. 15. 16. 15.  4. 16.  4. 12. 11. 14.  1. 14.
            # 16. 17. 16. 16. 16. 17. 16. 17. 15. 15.  8. 14. 12.  6. 16. 17. 17. 14.
            # 16.  9.  1. 10. 16. 17.  5.  7.  9.  7. 12. 10. 22.  8.  6. 10.  6. 17.
            #   7.  6. 10. 22. 12. 12.  1. 18.  9. 10.  1.  4.  4. 22.  7. 23.  1. 22.
            # 10. 22. 12. 15. 10. 10.  1.  8. 12. 10. 10. 10.  4.  4. 10.  9.  8. 11.
            # 17.  4. 18. 14. 22. 12. 12. 10. 14. 22. 10.  9. 16.  7. 12. 22.  6. 12.
            #   9.  7. 17.  7. 15. 20.  1.  5. 10. 12. 23.  7. 22. 12.  5. 23.  8. 14.
            #   6.  6.  4. 12.  7. 23.  9. 12. 15. 18.  4. 18.  2.  7. 12.  4. 16. 18.
            #   1. 18.  7. 23.  1.  1.  4.  8.  1. 17.  1. 12.  6. 23. 12.  9. 17. 12.
            # 14.  8. 22. 23. 18.  5. 12. 10. 16.  4. 14.  8. 12. 12.  5. 17. 15.  6.
            #   9.  8. 22. 12. 20.  5.  7.  6. 12.  2. 10.  4. 14.  8.  4.  9. 14. 17.
            # 20. 20. 17. 20.  7.  2.  6. 20. 16.  4. 23. 12.  1. 16.  4. 22. 12.  9.
            # 10. 15. 16. 23.  5. 18. 20. 22.  5. 21. 21.  2. 12.  6.  1. 22. 12. 23.
            #   4. 11. 12. 23.  6. 12. 16. 16.  7. 23. 17. 18. 14. 20.  1.  4. 11. 23.
            #   5. 14.  8. 12. 14. 10. 16. 12. 15. 12. 10. 16. 14.  8. 20. 14. 17.  1.
            # 18.  9. 12. 12.  4.  6. 12. 12. 15. 10.  5. 18. 20. 10. 18. 12.  6.  1.
            # 10.  5.  5.  7. 11.  4.  8. 10. 17.  9. 22.  7.  7. 17. 12.  8. 14. 21.
            #   5.  5. 12.  8. 14. 10. 10.  8. 20. 16. 16.  7. 14. 16. 14. 22. 17.  6.
            # 16. 14.  5. 12. 18.  9.  7.  5. 16.  4.  7. 20.  4.  4. 16. 22. 16. 14.
            # 17.  8.  6. 10. 17. 16.  5. 17. 22. 12. 22. 20.  1. 17. 22. 10.  4.  1.
            #   1.  4.  1.  7. 12.  6.  7. 17. 23.  1. 16. 16. 20.  5. 17.  7. 12.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
            #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    else:
        for d in ligands.keys():
            XD.append(one_hot_smiles(ligands[d], self.SMILEN, self.charsmiset))

        for t in proteins.keys():
            XT.append(one_hot_sequence(proteins[t], self.SEQLEN, self.charseqset))

    # ================================================================================
    int_based_label_of_all_smiles_sequences_np=np.asarray(int_based_label_of_all_smiles_sequences)
    int_based_label_of_all_protein_sequences_np=np.asarray(int_based_label_of_all_protein_sequences)
    smile_protein_binding_affinity_score_data_np=np.asarray(smile_protein_binding_affinity_score_data)

    # ================================================================================
    # @ Calculate number of drug and number of protein

    number_of_drugs=int_based_label_of_all_smiles_sequences_np.shape[0]
    number_of_proteins=int_based_label_of_all_protein_sequences_np.shape[0]
    # print(number_of_drugs)
    # print(number_of_proteins)
    # 2111
    # 229

    args.__setattr__("number_of_drugs",number_of_drugs)
    args.__setattr__("number_of_proteins",number_of_proteins)

    # ================================================================================
    mask_array_true_where_there_is_no_nan=np.isnan(smile_protein_binding_affinity_score_data_np)==False
    # print("mask_array_true_where_there_is_no_nan",mask_array_true_where_there_is_no_nan)
    # print("mask_array_true_where_there_is_no_nan",mask_array_true_where_there_is_no_nan.shape)
    # [[ True False False ... False False False]
    #  [ True False False ... False False False]
    #  [ True  True False ... False  True  True]
    #  ...
    #  [False False False ... False False  True]
    #  [False False False ... False False False]
    #  [False False False ... False False False]]
    # mask_array_true_where_there_is_no_nan (2111, 229)

    after_where=np.where(mask_array_true_where_there_is_no_nan)
    # print("after_where",after_where)
    # (array([   0,    0,    0, ..., 2110, 2110, 2110]), array([  0,   6,   9, ..., 156, 173, 223]))
    # (0,0) is true
    # (0,6) is true
    # ...

    label_row_inds,label_col_inds= np.where(np.isnan(after_where)==False)

    # ================================================================================
    # if not os.path.exists(figdir):
    #     os.makedirs(figdir)

    # print(FLAGS.log_dir)
    # logs/1578804744.5187547/

    # ================================================================================
    S1_avgperf, S1_avgloss, S1_teststd = nfold_1_2_3_setting_sample(
      int_based_label_of_all_smiles_sequences_np,int_based_label_of_all_protein_sequences_np,
      smile_protein_binding_affinity_score_data_np,label_row_inds,label_col_inds,args)

  # ================================================================================
  def __len__(self):
    # return self.nb_trn_imgs
    pass

  # ================================================================================
  def __getitem__(self,idx):
    # one_pair=self.trn_pairs[idx]
    # return one_pair
    pass
