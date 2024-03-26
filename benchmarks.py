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
      name='Imaging-clean',
      data_csv=PosixPath('./ScaledData/imaging_clean/pib_roi_annotated_clean.csv'),
      prediction_dir=PosixPath('./Predictions/Imaging data clean'),
      states = ['dvr_precentral_l', 'dvr_precentral_r', 'dvr_frontal_sup_l', 'dvr_frontal_sup_r', 'pet_date_mri_date_diff_days', 'dvr_frontal_sup_orb_l', 'dvr_frontal_sup_orb_r', 'dvr_frontal_mid_l', 'dvr_frontal_mid_r', 'dvr_frontal_mid_orb_l', 'dvr_frontal_mid_orb_r', 'dvr_frontal_inf_oper_l', 'dvr_frontal_inf_oper_r', 'dvr_frontal_inf_tri_l', 'dvr_frontal_inf_tri_r', 'dvr_frontal_inf_orb_l', 'dvr_frontal_inf_orb_r', 'dvr_rolandic_oper_l', 'dvr_rolandic_oper_r', 'dvr_supp_motor_area_l', 'dvr_supp_motor_area_r', 'dvr_olfactory_l', 'dvr_olfactory_r', 'dvr_frontal_sup_medial_l', 'dvr_frontal_sup_medial_r', 'dvr_frontal_med_orb_l', 'dvr_frontal_med_orb_r', 'dvr_rectus_l', 'dvr_rectus_r', 'dvr_insula_l', 'dvr_insula_r', 'dvr_cingulum_ant_l', 'dvr_cingulum_ant_r', 'dvr_cingulum_mid_l', 'dvr_cingulum_mid_r', 'dvr_cingulum_post_l', 'dvr_cingulum_post_r', 'dvr_hippocampus_l', 'dvr_hippocampus_r', 'dvr_parahippocampal_l', 'dvr_parahippocampal_r', 'dvr_amygdala_l', 'dvr_amygdala_r', 'dvr_calcarine_l', 'dvr_calcarine_r', 'dvr_cuneus_l', 'dvr_cuneus_r', 'dvr_lingual_l', 'dvr_lingual_r', 'dvr_occipital_sup_l', 'dvr_occipital_sup_r', 'dvr_occipital_mid_l', 'dvr_occipital_mid_r', 'dvr_occipital_inf_l', 'dvr_occipital_inf_r', 'dvr_fusiform_l', 'dvr_fusiform_r', 'dvr_postcentral_l', 'dvr_postcentral_r', 'dvr_parietal_sup_l', 'dvr_parietal_sup_r', 'dvr_parietal_inf_l', 'dvr_parietal_inf_r', 'dvr_supramarginal_l', 'dvr_supramarginal_r', 'dvr_angular_l', 'dvr_angular_r', 'dvr_precuneus_l', 'dvr_precuneus_r', 'dvr_paracentral_lobule_l', 'dvr_paracentral_lobule_r', 'dvr_caudate_l', 'dvr_caudate_r', 'dvr_putamen_l', 'dvr_putamen_r', 'dvr_pallidum_l', 'dvr_pallidum_r', 'dvr_thalamus_l', 'dvr_thalamus_r', 'dvr_heschl_l', 'dvr_heschl_r', 'dvr_temporal_sup_l', 'dvr_temporal_sup_r', 'dvr_temporal_pole_sup_l', 'dvr_temporal_pole_sup_r', 'dvr_temporal_mid_l', 'dvr_temporal_mid_r', 'dvr_temporal_pole_mid_l', 'dvr_temporal_pole_mid_r', 'dvr_temporal_inf_l', 'dvr_temporal_inf_r', 'dvr_cerebelum_crus1_l', 'dvr_cerebelum_crus1_r', 'dvr_cerebelum_crus2_l', 'dvr_cerebelum_crus2_r', 'dvr_cerebelum_3_l', 'dvr_cerebelum_3_r', 'dvr_cerebelum_4_5_l', 'dvr_cerebelum_4_5_r', 'dvr_cerebelum_6_l', 'dvr_cerebelum_6_r', 'dvr_cerebelum_7b_l', 'dvr_cerebelum_7b_r', 'dvr_cerebelum_8_l', 'dvr_cerebelum_8_r', 'dvr_cerebelum_9_l', 'dvr_cerebelum_9_r', 'dvr_cerebelum_10_l', 'dvr_cerebelum_10_r', 'dvr_vermis_1_2', 'dvr_vermis_3', 'dvr_vermis_4_5', 'dvr_vermis_6', 'dvr_vermis_7', 'dvr_vermis_8', 'dvr_vermis_9', 'dvr_vermis_10'],
      groups=['Train', 'Validate', 'Test'],
      time='pib_age',
      traj='wrapno'
  ),
]