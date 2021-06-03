""" Constants """

JOIN_ID = '#_join'
MERGE_ID = '#_merge'
REBATCH_ID = '#_rebatch'
PIPELINE_ID = '#_pipeline'
IMPORT_MODEL_ID = '#_import_model'
TRAIN_MODEL_ID = '#_train_model'
PREDICT_MODEL_ID = '#_predict_model'
SAVE_MODEL_ID = '#_save_model'
LOAD_MODEL_ID = '#_load_model'
GATHER_METRICS_ID = '#_gather_metrics'
UPDATE_VARIABLE_ID = '#_update_variable'
UPDATE_ID = '#_update'
CALL_ID = '#_call'
PRINT_ID = '#_print'
CALL_FROM_NS_ID = '#_from_ns'
ACQUIRE_LOCK_ID = '#_acquire_lock'
RELEASE_LOCK_ID = '#_release_lock'
DISCARD_BATCH_ID = '#_discard_batch'

ACTIONS = {
    IMPORT_MODEL_ID: '_exec_import_model',
    TRAIN_MODEL_ID: '_exec_train_model',
    PREDICT_MODEL_ID: '_exec_predict_model',
    SAVE_MODEL_ID: '_exec_save_model',
    LOAD_MODEL_ID: '_exec_load_model',
    GATHER_METRICS_ID: '_exec_gather_metrics',
    UPDATE_VARIABLE_ID: '_exec_update_variable',
    UPDATE_ID: '_exec_update',
    CALL_ID: '_exec_call',
    PRINT_ID: '_exec_print',
    CALL_FROM_NS_ID: '_exec_from_ns',
    ACQUIRE_LOCK_ID: '_exec_acquire_lock',
    RELEASE_LOCK_ID: '_exec_release_lock',
}
