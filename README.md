Инструкция для запуска:
> // argv[1] - путь к матрице  
> // argv[2] - путь к файлу, куда запишется лог  
> // argv[3] - режим запуска (tobinary - конвертация из .mtx в .bin / launch - запуск алгоритма)  
> // в случае 'launch':  
> //   argv[4] - запускаемый алгоритм (triangle / k-truss / mxm / bc)  
> //   argv[5] - последовательный / параллельный запуск (seq / par)  
> //   в случае 'argv[4] == bc':  
> //     argv[6] - batch size (количество стартовых вершин, которые обрабатываем в алгоритме)  
> //   в случае 'argv[4] == triangle' / 'k-truss' / 'mxm':  
> //     argv[6] - вид алгоритма умножения матриц (naive / msa / mca / heap)  
> //     также в случае 'argv[4] == k-truss':  
> //       argv[6] - параметр 'k' в k-truss    
  
Пример запуска:
> ./grAlgo matrix.mtx log.txt launch mxm par
