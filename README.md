In order to run our scripts, you first need to download the raw MultiNews data from this [this link](https://github.com/Alex-Fabbri/Multi-News).

Afterwards you can run ./scripts/preprocess_multinews.sh, where you can modify the desired parameters.

Then you can start training the GraphSum model on sentence and paragraph level with ./scripts/run_graphsum_local_multinews_sentence.sh and ./scripts/run_graphsum_local_multinews_paragraphs.sh respectively. The obtained rouge scores can then be found in the log folder.

Further documentation can be found in the code or our technical report.
