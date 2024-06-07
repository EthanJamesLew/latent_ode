import sysidexpr.model as sidmodel
import sysidexpr.constant as sidconst
import sysidexpr.benchmark as sidbench
import sysidexpr.loss as sidloss

from pathlib import PosixPath

benchmarks = [
    sidbench.BenchmarkConfiguration(
      name='FHN',
      data_csv=PosixPath('./ScaledData/fhn/FHN_data_1.csv'),
      prediction_dir=PosixPath('./Predictions/FHN data'),
      states=['X1','X2'],
      groups=['Train', 'Validate', 'Test'],
      time='t',
      traj='id'
    ),
    sidbench.BenchmarkConfiguration(
      name='Lorenz96_1024',
      data_csv=PosixPath('./ScaledData/lorenz96_1024/lorenz96_1024_annotated.csv'),
      prediction_dir=PosixPath('./Predictions/Lorenz96-1024'),
      states = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14',
                'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X24', 'X25', 'X26', 'X27',
                'X28', 'X29', 'X30', 'X31', 'X32', 'X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39', 'X40',
                'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49', 'X50', 'X51', 'X52', 'X53',
                'X54', 'X55', 'X56', 'X57', 'X58', 'X59', 'X60', 'X61', 'X62', 'X63', 'X64', 'X65', 'X66', 'X67',
                'X68', 'X69', 'X70', 'X71', 'X72', 'X73', 'X74', 'X75', 'X76', 'X77', 'X78', 'X79', 'X80', 'X81',
                'X82', 'X83', 'X84', 'X85', 'X86', 'X87', 'X88', 'X89', 'X90', 'X91', 'X92', 'X93', 'X94', 'X95',
                'X96', 'X97', 'X98', 'X99', 'X100', 'X101', 'X102', 'X103', 'X104', 'X105', 'X106', 'X107', 'X108',
                'X109', 'X110', 'X111', 'X112', 'X113', 'X114', 'X115', 'X116', 'X117', 'X118', 'X119', 'X120', 'X121',
                'X122', 'X123', 'X124', 'X125', 'X126', 'X127', 'X128', 'X129', 'X130', 'X131', 'X132', 'X133', 'X134',
                'X135', 'X136', 'X137', 'X138', 'X139', 'X140', 'X141', 'X142', 'X143', 'X144', 'X145', 'X146', 'X147',
                'X148', 'X149', 'X150', 'X151', 'X152', 'X153', 'X154', 'X155', 'X156', 'X157', 'X158', 'X159', 'X160',
                'X161', 'X162', 'X163', 'X164', 'X165', 'X166', 'X167', 'X168', 'X169', 'X170', 'X171', 'X172', 'X173',
                'X174', 'X175', 'X176', 'X177', 'X178', 'X179', 'X180', 'X181', 'X182', 'X183', 'X184', 'X185', 'X186', 'X187', 'X188', 'X189', 'X190', 'X191', 'X192', 'X193', 'X194', 'X195', 'X196', 'X197', 'X198', 'X199', 'X200', 'X201', 'X202', 'X203', 'X204', 'X205', 'X206', 'X207', 'X208', 'X209', 'X210', 'X211', 'X212', 'X213', 'X214', 'X215', 'X216', 'X217', 'X218', 'X219', 'X220', 'X221', 'X222', 'X223', 'X224', 'X225', 'X226', 'X227', 'X228', 'X229', 'X230', 'X231', 'X232', 'X233', 'X234', 'X235', 'X236', 'X237', 'X238', 'X239', 'X240', 'X241', 'X242', 'X243', 'X244', 'X245', 'X246', 'X247', 'X248', 'X249', 'X250', 'X251', 'X252', 'X253', 'X254', 'X255', 'X256', 'X257', 'X258', 'X259', 'X260', 'X261', 'X262', 'X263', 'X264', 'X265', 'X266', 'X267', 'X268', 'X269', 'X270', 'X271', 'X272', 'X273', 'X274', 'X275', 'X276', 'X277', 'X278', 'X279', 'X280', 'X281', 'X282', 'X283', 'X284', 'X285', 'X286', 'X287', 'X288', 'X289', 'X290', 'X291', 'X292', 'X293', 'X294', 'X295', 'X296', 'X297', 'X298', 'X299', 'X300', 'X301', 'X302', 'X303', 'X304', 'X305', 'X306', 'X307', 'X308', 'X309', 'X310', 'X311', 'X312', 'X313', 'X314', 'X315', 'X316', 'X317', 'X318', 'X319', 'X320', 'X321', 'X322', 'X323', 'X324', 'X325', 'X326', 'X327', 'X328', 'X329', 'X330', 'X331', 'X332', 'X333', 'X334', 'X335', 'X336', 'X337', 'X338', 'X339', 'X340', 'X341', 'X342', 'X343', 'X344', 'X345', 'X346', 'X347', 'X348', 'X349', 'X350', 'X351', 'X352', 'X353', 'X354', 'X355', 'X356', 'X357', 'X358', 'X359', 'X360', 'X361', 'X362', 'X363', 'X364', 'X365', 'X366', 'X367', 'X368', 'X369', 'X370', 'X371', 'X372', 'X373', 'X374', 'X375', 'X376', 'X377', 'X378', 'X379', 'X380', 'X381', 'X382', 'X383', 'X384', 'X385', 'X386', 'X387', 'X388', 'X389', 'X390', 'X391', 'X392', 'X393', 'X394', 'X395', 'X396', 'X397', 'X398', 'X399', 'X400', 'X401', 'X402', 'X403', 'X404', 'X405', 'X406', 'X407', 'X408', 'X409', 'X410', 'X411', 'X412', 'X413', 'X414', 'X415', 'X416', 'X417', 'X418', 'X419', 'X420', 'X421', 'X422', 'X423', 'X424', 'X425', 'X426', 'X427', 'X428', 'X429', 'X430', 'X431', 'X432', 'X433', 'X434', 'X435', 'X436', 'X437', 'X438', 'X439', 'X440', 'X441', 'X442', 'X443', 'X444', 'X445', 'X446', 'X447', 'X448', 'X449', 'X450', 'X451', 'X452', 'X453', 'X454', 'X455', 'X456', 'X457', 'X458', 'X459', 'X460', 'X461', 'X462', 'X463', 'X464', 'X465', 'X466', 'X467', 'X468', 'X469', 'X470', 'X471', 'X472', 'X473', 'X474', 'X475', 'X476', 'X477', 'X478', 'X479', 'X480', 'X481', 'X482', 'X483', 'X484', 'X485', 'X486', 'X487', 'X488', 'X489', 'X490', 'X491', 'X492', 'X493', 'X494', 'X495', 'X496', 'X497', 'X498', 'X499', 'X500', 'X501', 'X502', 'X503', 'X504', 'X505', 'X506', 'X507', 'X508', 'X509', 'X510', 'X511', 'X512', 'X513', 'X514', 'X515', 'X516', 'X517', 'X518', 'X519', 'X520', 'X521', 'X522', 'X523', 'X524', 'X525', 'X526', 'X527', 'X528', 'X529', 'X530', 'X531', 'X532', 'X533', 'X534', 'X535', 'X536', 'X537', 'X538', 'X539', 'X540', 'X541', 'X542', 'X543', 'X544', 'X545', 'X546', 'X547', 'X548', 'X549', 'X550', 'X551', 'X552', 'X553', 'X554', 'X555', 'X556', 'X557', 'X558', 'X559', 'X560', 'X561', 'X562', 'X563', 'X564', 'X565', 'X566', 'X567', 'X568', 'X569', 'X570', 'X571', 'X572', 'X573', 'X574', 'X575', 'X576', 'X577', 'X578', 'X579', 'X580', 'X581', 'X582', 'X583', 'X584', 'X585', 'X586', 'X587', 'X588', 'X589', 'X590', 'X591', 'X592', 'X593', 'X594', 'X595', 'X596', 'X597', 'X598', 'X599', 'X600', 'X601', 'X602', 'X603', 'X604', 'X605', 'X606', 'X607', 'X608', 'X609', 'X610', 'X611', 'X612', 'X613', 'X614', 'X615', 'X616', 'X617', 'X618', 'X619', 'X620', 'X621', 'X622', 'X623', 'X624', 'X625', 'X626', 'X627', 'X628', 'X629', 'X630', 'X631', 'X632', 'X633', 'X634', 'X635', 'X636', 'X637', 'X638', 'X639', 'X640', 'X641', 'X642', 'X643', 'X644', 'X645', 'X646', 'X647', 'X648', 'X649', 'X650', 'X651', 'X652', 'X653', 'X654', 'X655', 'X656', 'X657', 'X658', 'X659', 'X660', 'X661', 'X662', 'X663', 'X664', 'X665', 'X666', 'X667', 'X668', 'X669', 'X670', 'X671', 'X672', 'X673', 'X674', 'X675', 'X676', 'X677', 'X678', 'X679', 'X680', 'X681', 'X682', 'X683', 'X684', 'X685', 'X686', 'X687', 'X688', 'X689', 'X690', 'X691', 'X692', 'X693', 'X694', 'X695', 'X696', 'X697', 'X698', 'X699', 'X700', 'X701', 'X702', 'X703', 'X704', 'X705', 'X706', 'X707', 'X708', 'X709', 'X710', 'X711', 'X712', 'X713', 'X714', 'X715', 'X716', 'X717', 'X718', 'X719', 'X720', 'X721', 'X722', 'X723', 'X724', 'X725', 'X726', 'X727', 'X728', 'X729', 'X730', 'X731', 'X732', 'X733', 'X734', 'X735', 'X736', 'X737', 'X738', 'X739', 'X740', 'X741', 'X742', 'X743', 'X744', 'X745', 'X746', 'X747', 'X748', 'X749', 'X750', 'X751', 'X752', 'X753', 'X754', 'X755', 'X756', 'X757', 'X758', 'X759', 'X760', 'X761', 'X762', 'X763', 'X764', 'X765', 'X766', 'X767', 'X768', 'X769', 'X770', 'X771', 'X772', 'X773', 'X774', 'X775', 'X776', 'X777', 'X778', 'X779', 'X780', 'X781', 'X782', 'X783', 'X784', 'X785', 'X786', 'X787', 'X788', 'X789', 'X790', 'X791', 'X792', 'X793', 'X794', 'X795', 'X796', 'X797', 'X798', 'X799', 'X800', 'X801', 'X802', 'X803', 'X804', 'X805', 'X806', 'X807', 'X808', 'X809', 'X810', 'X811', 'X812', 'X813', 'X814', 'X815', 'X816', 'X817', 'X818', 'X819', 'X820', 'X821', 'X822', 'X823', 'X824', 'X825', 'X826', 'X827', 'X828', 'X829', 'X830', 'X831', 'X832', 'X833', 'X834', 'X835', 'X836', 'X837', 'X838', 'X839', 'X840', 'X841', 'X842', 'X843', 'X844', 'X845', 'X846', 'X847', 'X848', 'X849', 'X850', 'X851', 'X852', 'X853', 'X854', 'X855', 'X856', 'X857', 'X858', 'X859', 'X860', 'X861', 'X862', 'X863', 'X864', 'X865', 'X866', 'X867', 'X868', 'X869', 'X870', 'X871', 'X872', 'X873', 'X874', 'X875', 'X876', 'X877', 'X878', 'X879', 'X880', 'X881', 'X882', 'X883', 'X884', 'X885', 'X886', 'X887', 'X888', 'X889', 'X890', 'X891', 'X892', 'X893', 'X894', 'X895', 'X896', 'X897', 'X898', 'X899', 'X900', 'X901', 'X902', 'X903', 'X904', 'X905', 'X906', 'X907', 'X908', 'X909', 'X910', 'X911', 'X912', 'X913', 'X914', 'X915', 'X916', 'X917', 'X918', 'X919', 'X920', 'X921', 'X922', 'X923', 'X924', 'X925', 'X926', 'X927', 'X928', 'X929', 'X930', 'X931', 'X932', 'X933', 'X934', 'X935', 'X936', 'X937', 'X938', 'X939', 'X940', 'X941', 'X942', 'X943', 'X944', 'X945', 'X946', 'X947', 'X948', 'X949', 'X950', 'X951', 'X952', 'X953', 'X954', 'X955', 'X956', 'X957', 'X958', 'X959', 'X960', 'X961', 'X962', 'X963', 'X964', 'X965', 'X966', 'X967', 'X968', 'X969', 'X970', 'X971', 'X972', 'X973', 'X974', 'X975', 'X976', 'X977', 'X978', 'X979', 'X980', 'X981', 'X982', 'X983', 'X984', 'X985', 'X986', 'X987', 'X988', 'X989', 'X990', 'X991', 'X992', 'X993', 'X994', 'X995', 'X996', 'X997', 'X998', 'X999', 'X1000', 'X1001', 'X1002', 'X1003', 'X1004', 'X1005', 'X1006', 'X1007', 'X1008', 'X1009', 'X1010', 'X1011', 'X1012', 'X1013', 'X1014', 'X1015', 'X1016', 'X1017', 'X1018', 'X1019', 'X1020', 'X1021', 'X1022', 'X1023', 'X1024'],
      groups=['Train', 'Validate', 'Test'],
      time='t',
      traj='id'
  ),
    sidbench.BenchmarkConfiguration(
      name='Lorenz96_128',
      data_csv=PosixPath('./ScaledData/lorenz96_128/lorenz96_128_annotated.csv'),
      prediction_dir=PosixPath('./Predictions/Lorenz96-128'),
      states = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14',
                 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X24', 'X25', 'X26',
                 'X27', 'X28', 'X29', 'X30', 'X31', 'X32', 'X33', 'X34', 'X35', 'X36', 'X37', 'X38',
                 'X39', 'X40', 'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49', 'X50',
                 'X51', 'X52', 'X53', 'X54', 'X55', 'X56', 'X57', 'X58', 'X59', 'X60', 'X61', 'X62',
                 'X63', 'X64', 'X65', 'X66', 'X67', 'X68', 'X69', 'X70', 'X71', 'X72', 'X73', 'X74',
                 'X75', 'X76', 'X77', 'X78', 'X79', 'X80', 'X81', 'X82', 'X83', 'X84', 'X85', 'X86',
                 'X87', 'X88', 'X89', 'X90', 'X91', 'X92', 'X93', 'X94', 'X95', 'X96', 'X97', 'X98',
                 'X99', 'X100', 'X101', 'X102', 'X103', 'X104', 'X105', 'X106', 'X107', 'X108', 'X109',
                 'X110', 'X111', 'X112', 'X113', 'X114', 'X115', 'X116', 'X117', 'X118', 'X119', 'X120',
                 'X121', 'X122', 'X123', 'X124', 'X125', 'X126', 'X127', 'X128'],
      groups=['Train', 'Validate', 'Test'],
      time='time',
      traj='id'
  ),
  sidbench.BenchmarkConfiguration(
      name='Lorenz96_32',
      data_csv=PosixPath('./ScaledData/lorenz96_32/lorenz96_32_annotated.csv'),
      prediction_dir=PosixPath('./Predictions/Lorenz96-32'),
      states = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14',
                'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X24', 'X25', 'X26',
                'X27', 'X28', 'X29', 'X30', 'X31', 'X32'],
      groups=['Train', 'Validate', 'Test'],
      time='time',
      traj='id'
  ),
  sidbench.BenchmarkConfiguration(
      name='Lorenz96_16',
      data_csv=PosixPath('./ScaledData/lorenz96_16/lorenz96_16_annotated.csv'),
      prediction_dir=PosixPath('./Predictions/Lorenz96-16'),
      states = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8',
                'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16'],
      groups=['Train', 'Validate', 'Test'],
      time='time',
      traj='id'
  ),
  sidbench.BenchmarkConfiguration(
      name='Lorenz63',
      data_csv=PosixPath('./ScaledData/lorenz/Lorenz_data_1.csv'),
      prediction_dir=PosixPath('./Predictions/Lorenz data'),
      states = ['X1', 'X2', 'X3'],
      groups=['Train', 'Validate', 'Test'],
      time='t',
      traj='id'
  ),
    sidbench.BenchmarkConfiguration(
      name='Imaging-clean',
      data_csv=PosixPath('./ScaledData/imaging_clean/pib_roi_annotated_clean.csv'),
      prediction_dir=PosixPath('./Predictions/Imaging data clean'),
      states = ['dvr_precentral_l', 'dvr_precentral_r', 'dvr_frontal_sup_l', 'dvr_frontal_sup_r', 'pet_date_mri_date_diff_days', 'dvr_frontal_sup_orb_l', 'dvr_frontal_sup_orb_r', 'dvr_frontal_mid_l', 'dvr_frontal_mid_r', 'dvr_frontal_mid_orb_l', 'dvr_frontal_mid_orb_r', 'dvr_frontal_inf_oper_l', 'dvr_frontal_inf_oper_r', 'dvr_frontal_inf_tri_l', 'dvr_frontal_inf_tri_r', 'dvr_frontal_inf_orb_l', 'dvr_frontal_inf_orb_r', 'dvr_rolandic_oper_l', 'dvr_rolandic_oper_r', 'dvr_supp_motor_area_l', 'dvr_supp_motor_area_r', 'dvr_olfactory_l', 'dvr_olfactory_r', 'dvr_frontal_sup_medial_l', 'dvr_frontal_sup_medial_r', 'dvr_frontal_med_orb_l', 'dvr_frontal_med_orb_r', 'dvr_rectus_l', 'dvr_rectus_r', 'dvr_insula_l', 'dvr_insula_r', 'dvr_cingulum_ant_l', 'dvr_cingulum_ant_r', 'dvr_cingulum_mid_l', 'dvr_cingulum_mid_r', 'dvr_cingulum_post_l', 'dvr_cingulum_post_r', 'dvr_hippocampus_l', 'dvr_hippocampus_r', 'dvr_parahippocampal_l', 'dvr_parahippocampal_r', 'dvr_amygdala_l', 'dvr_amygdala_r', 'dvr_calcarine_l', 'dvr_calcarine_r', 'dvr_cuneus_l', 'dvr_cuneus_r', 'dvr_lingual_l', 'dvr_lingual_r', 'dvr_occipital_sup_l', 'dvr_occipital_sup_r', 'dvr_occipital_mid_l', 'dvr_occipital_mid_r', 'dvr_occipital_inf_l', 'dvr_occipital_inf_r', 'dvr_fusiform_l', 'dvr_fusiform_r', 'dvr_postcentral_l', 'dvr_postcentral_r', 'dvr_parietal_sup_l', 'dvr_parietal_sup_r', 'dvr_parietal_inf_l', 'dvr_parietal_inf_r', 'dvr_supramarginal_l', 'dvr_supramarginal_r', 'dvr_angular_l', 'dvr_angular_r', 'dvr_precuneus_l', 'dvr_precuneus_r', 'dvr_paracentral_lobule_l', 'dvr_paracentral_lobule_r', 'dvr_caudate_l', 'dvr_caudate_r', 'dvr_putamen_l', 'dvr_putamen_r', 'dvr_pallidum_l', 'dvr_pallidum_r', 'dvr_thalamus_l', 'dvr_thalamus_r', 'dvr_heschl_l', 'dvr_heschl_r', 'dvr_temporal_sup_l', 'dvr_temporal_sup_r', 'dvr_temporal_pole_sup_l', 'dvr_temporal_pole_sup_r', 'dvr_temporal_mid_l', 'dvr_temporal_mid_r', 'dvr_temporal_pole_mid_l', 'dvr_temporal_pole_mid_r', 'dvr_temporal_inf_l', 'dvr_temporal_inf_r', 'dvr_cerebelum_crus1_l', 'dvr_cerebelum_crus1_r', 'dvr_cerebelum_crus2_l', 'dvr_cerebelum_crus2_r', 'dvr_cerebelum_3_l', 'dvr_cerebelum_3_r', 'dvr_cerebelum_4_5_l', 'dvr_cerebelum_4_5_r', 'dvr_cerebelum_6_l', 'dvr_cerebelum_6_r', 'dvr_cerebelum_7b_l', 'dvr_cerebelum_7b_r', 'dvr_cerebelum_8_l', 'dvr_cerebelum_8_r', 'dvr_cerebelum_9_l', 'dvr_cerebelum_9_r', 'dvr_cerebelum_10_l', 'dvr_cerebelum_10_r', 'dvr_vermis_1_2', 'dvr_vermis_3', 'dvr_vermis_4_5', 'dvr_vermis_6', 'dvr_vermis_7', 'dvr_vermis_8', 'dvr_vermis_9', 'dvr_vermis_10'],
      groups=['Train', 'Validate', 'Test'],
      time='pib_age',
      traj='wrapno'
  ),
]

current_benchmark = benchmarks[-1]