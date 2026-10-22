Тестовые матрицы: https://drive.google.com/file/d/1IGwGVefNt90q9l977yTW3p0unQdssD_o/view?usp=sharing  
Их стоит поместить в папку graphs, чтобы работали тесты make mxm_all_test и остальные тесты из Makefile

Инструкция для сборки:  
> make - стандартная сборка  
> make rvv - RISC-V сборка с MCA_LMUL=1 (кросскомпиляция)  
> make rvv-native - RISC-V сборка с MCA_LMUL=1 на RISC-V узле  
> make rvv|rvv-native MCA_LMUL=1|2|4 - RISC-V сборка с MCA_LMUL=1,2,4  

Инструкция для запуска:
> make test - собрать проект, запустить проверки корректности и затем бенчмарки mxm, triangle и k-truss на netherlands_osm.mtx  
> make test_rvv - собрать с RVV на RISC-V узле, запустить проверки корректности и бенчмарки  
> ./build/mxm_test ./graphs/netherlands_osm.mtx log.txt - запустить тест умножения матриц  
> ./build/triangle_test ./graphs/netherlands_osm.mtx log.txt - запустить тест подсчёта треугольников  
> ./build/k_truss_test ./graphs/netherlands_osm.mtx log.txt 3 - запустить тест k-truss (последний аргумент — значение k)  
> make mxm_test|triangle_test|k_truss_test - запустить набор тестов на SMALL_GRAPHS из Makefile  
> make mxm_all_test|triangle_all_test|k_truss_all_test - запустить набор тестов на ALL_GRAPHS из Makefile
