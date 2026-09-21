Fluxo:

Anurandenoising pega os audios originais e gera os audios limpos com trimming
cnn_optimization e panns_optimization: otimizam seus parametros com Optuna para maximizar a acuracia no LOOCV
cnn e panns: rodam os modelos otimizados para 15 sementes
