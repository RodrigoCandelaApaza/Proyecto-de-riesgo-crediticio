# Proyecto de riesgo crediticio
Primer proyecto de práctica de Data Science. Uso modelos de clasificación para determinar qué clientes de un banco ficticio pueden ser incumplidores utilizando sus características como variables explicativas. Obtengo un ajuste de 0,94 en el grupo de prueba con un modelo Gradient Boosting.
 ## 1. Análisis
 Contamos con información de 148 670 clientes del Banco y, para cada uno, variables como género, límite del préstamo, crédito del cliente, tasas de interés, plazo del préstamo, entre otras. Podemos extraer algunos insights clave:
 
![image](https://github.com/RodrigoCandelaApaza/Proyecto-de-riesgo-crediticio/assets/58021217/b24e0028-ae88-4d73-b4f9-69a07ade845e) \
La gran mayoría de clientes (cerca de 60 000) han solicitado créditos de hasta 250 mil USD, mientras que los demás solicitaron créditos por montos superiores. 

![image](https://github.com/RodrigoCandelaApaza/Proyecto-de-riesgo-crediticio/assets/58021217/e6684b8e-53df-45c0-8d8b-474115c4c89f) \
La puntuación crediticia sigue una distribución uniforme, por lo que quizás no es muy útil como variable explicativa.

![image](https://github.com/RodrigoCandelaApaza/Proyecto-de-riesgo-crediticio/assets/58021217/b5465e80-17ca-4d7e-817d-ee70c65e2322) \
Los clientes clasificados como `l1` en la variable `Credith_Worthiness` cuentan con una tasa de interés menor, por lo que representan menor riesgo para el Banco.
 ## 2. Limpieza
 En esta sección analizo los outliers o valores extremos, y opto por eliminarlos de la base de datos, con lo que restan 15 202 observaciones. \
 A continuación, cambio los tipos de datos para cada variable tal que coincidan con los que les corresponden. Por ejemplo, `Credit_Worthiness` corresponde a una variable binaria que indica si un cliente tiene buen crédito (`l1`) o no (`l2`). Reemplazo `l1` por 1 y `l2` por 0. El mismo tratamiento se le da a `approv_in_adv`, `Neg_ammortization`, `interest_only` y `lump_sum_payment`. Para variables categoricas utilizo el método `get_dummies()` de pandas para obtener variables binarias.
 ## 3. Modelos 
 La variable dependiente que usaré será `Credit_Worthiness`, donde 1 equivale a buen cliente y 0 a posible deudor. Por este motivo, el problema que tratamos de resolver es de clasificación. Sin embargo, esta variable presenta 14 744 buenos pagadores y solo 458 posibles deudores, por lo que un modelo de clasificación estará sesgado y posiblemente clasifique como buen cliente a un posible deudor. Para balancear la variable, se crean observaciones sintéticas utilizando las disponibles como referencia mediante el método SMOTE. \
 \
 Una vez balanceada la variable dependiente, se divide la muestra en un conjunto de entrenamiento (70%) y otro de prueba (30%). Utilizamos como modelo base una regresión logística y otro modelo candidato de Gradient Boosting. El modelo de regresión logística consiguió un ajuste de 0.559 y el de Gradient Boosting un ajuste de 0.958, tras utilizar una validación cruzada con 5 dobleces y 100 iteraciones. \
 \
 Finalmente, la tasa de interés es la variable más importante para predecir si un cliente es digno de crédito o no.
