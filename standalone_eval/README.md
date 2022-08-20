mTVR Evalation
================================================================

mTVR uses the *exact* same evaluation script as [TVR evaluation](https://github.com/jayleicn/TVRetrieval/tree/master/standalone_eval), please refer to details there. 
Specifically, you should be able to find the following resources:

* Task Definition
* How to construct a prediction file?
* Run Evaluation

### Codalab Submission

To test your model's performance on test-public set, please submit both val and test-public predictions to our [Codalab evaluation server](https://codalab.lisn.upsaclay.fr/competitions/6973). The submission file should be a single .zip file (no enclosing folder) that contains the 4 prediction files `zh_tvr_val_submission.json`, `zh_tvr_test_public_submission.json`, `en_tvr_val_submission.json` and `en_tvr_test_public_submission.json`. Each of the `*submission.json` file should be formatted as instructed in [TVR Evaluation](https://github.com/jayleicn/TVRetrieval/tree/master/standalone_eval#how-to-construct-a-prediction-file). 
