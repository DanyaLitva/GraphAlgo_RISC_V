Инструкция для сборки:  
> make - стандартная сборка  
> make rvv - RISC-V сборка с LMUL=2 (кросскомпиляция)  
> make rvv-native - RISC-V сборка с LMUL=2 на RISC-V узле  
> make rvv|rvv-native LMUL=1|2|4 - RISC-V сборка с LMUL=1,2,4  

Инструкция для запуска:
> // argv[1] - путь к матрице  
> // argv[2] - путь к файлу, куда запишется лог  
> // argv[3] - режим запуска (tobinary - конвертация из .mtx в .bin / launch - запуск алгоритма)    
> // в случае 'launch':  
> //   argv[4] - запускаемый алгоритм (triangle / k-truss / mxm / bc)  
> //   argv[5] - последовательный / параллельный запуск (seq / par)  
> //   argv[6] - скалярное / rvv умножение (scal / vec)   
> //   в случае 'argv[4] == bc':  
> //     argv[7] - batch size (количество стартовых вершин, которые обрабатываем в алгоритме)  
> //   в случае 'argv[4] == triangle' / 'k-truss' / 'mxm':  
> //     argv[7] - вид алгоритма умножения матриц (naive / msa / mca / heap)  
> //     также в случае 'argv[4] == k-truss':  
> //       argv[8] - параметр 'k' в k-truss  

Пример запуска:
> ./build/grAlgo ./graphs/G43.mtx log.txt launch mxm par vec mca
