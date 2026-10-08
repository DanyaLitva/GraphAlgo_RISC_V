Актуальная векторизованная версия MCA: https://github.com/DanyaLitva/GraphAlgo_RISC_V/blob/3691adbc351f007b832a97c21863ca6a80d4dcb5/include/matrix_la.h#L226  

Актуальная векторизованная версия HEAP: https://github.com/DanyaLitva/GraphAlgo_RISC_V/blob/3691adbc351f007b832a97c21863ca6a80d4dcb5/include/matrix_la.h#L864  

Тестовые матрицы: https://drive.google.com/file/d/1IGwGVefNt90q9l977yTW3p0unQdssD_o/view?usp=sharing  
Их стоит поместить в папку graphs, чтобы работали тесты make mxm_all_test и остальные тесты из Makefile

Инструкция для сборки:  
> make - стандартная сборка  
> make rvv - RISC-V сборка с MCA_LMUL=1 (кросскомпиляция)  
> make rvv-native - RISC-V сборка с MCA_LMUL=1 на RISC-V узле  
> make rvv|rvv-native MCA_LMUL=1|2|4 - RISC-V сборка с MCA_LMUL=1,2,4  

Инструкция для запуска: TODO
