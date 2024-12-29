# Entregable-atm-sferas-II
Segundo entregable de Atmósferas Estelares sobre poblaciones y opacidades, por Eneko Lekaroz y Daniel Sosa

El programa toma como input los dos modelos en formato .dat. Previamente se han modificado los archivos originales para incluir solo las tablas finales.
Para usar el programa hay que modificar 'ruta_5000' y 'ruta_8000' a las rutas en las que se hayan guardado dichos ficheros.
También hay que modificar 'ruta' a la ruta en la que se deseen guardar los resultados.

En primer lugar el programa crea un diccionario con las tablas, para facilitar el resto de operaciones.
Después se plotean las diferentes variables de interes. Se calcula la temperatura de cuerpo gris para poder dibujarla.

La función Saha calcula las poblaciones de las diferentes especies ionizadas, tomando como input la temperatura y presión electrónica de las tablas.
Se definen en variables las energías y funciones de partición, así como las constantes físicas.
De la misma manera, la función Boltzmann toma la temperatura para dar las poblaciones de los estados excitados del H neutro-

Para dar con las T y Pe de input (a la tau deseada), se itera por las tablas hasta encontrar el valor de tau_R más cercano al pedido, y se guarda la temperatura y presión correspondientes.
Se llama a las funciones con esos valores. En el caso de Boltzmann hay que resolver el sistema de ecuaciones para obtener cada población.
Se guardan todos los resultados en dataframes, uno por modelo.

Para las poblaciones, se definen funciones para cada contribución, teniendo en cuenta que las ligado-libre solo son válidas hasta cierto umbral. Definimos dicho umbral. Definimos también dentro de las funciones los factores de Gaunt, secciones eficaces, etc.
Definimos también los cantos (como array) y se los pasamos a las funciones para obtener los resultados numéricos. Los guardamos en dataframes.
Ploteamos los resultados entre 500 y 20000 A.

Para terminar, creamos dos funciones para las líneas de Balmer y Lyman, donde se definen los niveles y los factores de Gaunt.
Se calculan las opacidades de las líneas.
