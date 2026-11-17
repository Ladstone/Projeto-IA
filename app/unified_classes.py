"""Rótulos visuais comuns aos datasets MyFood e Brazilian Food Images.

IDs visuais são distintos dos códigos TACO: a TACO descreve composição nutricional,
não anotações de imagens. Cada correspondência nutricional é uma HIPÓTESE editável.
"""
CLASSES={
 1:('Arroz',3,125,'Arroz branco tipo 1 cozido — preparo presumido'),
 2:('Feijão',567,110,'Feijão-preto cozido como referência; confirme o tipo'),
 3:('Frango',410,130,'Peito sem pele grelhado como aproximação; confirme o corte e preparo'),
 4:('Carne bovina',377,130,'Patinho sem gordura grelhado como aproximação'),
 5:('Salada',78,65,'Alface crespa crua como referência: misturas têm composição diferente'),
 6:('Batata frita',93,100,'Batata inglesa frita'),
 7:('Ovo',488,55,'Ovo cozido como referência; altere para ovo frito se necessário'),
 8:('Macarrão',542,150,'Macarrão à bolonhesa como aproximação; confirme molho'),
 9:('Maçã',222,130,'Maçã Fuji com casca crua como aproximação'),
 10:('Tomate',157,60,'Tomate com sementes cru como aproximação'),
 11:('Cenoura',109,65,'Cenoura cozida como aproximação'),
 12:('Abóbora',64,80,'Abóbora cabotiá cozida como aproximação'),
 13:('Bife à milanesa',340,130,'Contra-filé à milanesa como aproximação'),
 14:('Farofa',131,40,'Farofa de mandioca temperada como aproximação'),
 15:('Beterraba',97,60,'Beterraba cozida como aproximação'),
 16:('Purê',91,115,'Batata inglesa cozida como PROXY: purê com leite/manteiga tem outros nutrientes'),
 17:('Massa com molho vermelho',542,140,'Macarrão à bolonhesa como PROXY: molho vermelho pode não ter carne'),
 18:('Repolho',149,55,'Repolho branco cru como aproximação'),
 19:('Milho',45,70,'Milho verde enlatado e drenado como aproximação'),
 20:('Salada com maionese',545,90,'Salada de legumes com maionese como aproximação'),
 21:('Mandioca',129,100,'Mandioca cozida como aproximação'),
}
N_CLASS=len(CLASSES)
CLASS_LABELS={k:v[0] for k,v in CLASSES.items()}
# Mais de 20 cores: distinção visual, não graus de confiança.
PALETTE=['#70b8ee','#8f6e50','#edaa6b','#b96564','#6fbc92','#e5b84c','#ecce79','#bb90d5','#f27991','#e86d69','#ed8f4a','#e3a05b','#ad7760','#cfb97c','#c76992','#eac69a','#ce5a5a','#71b36d','#e9d353','#c8bda5','#a28c64']
LABEL_COLORS={k:PALETTE[k-1] for k in CLASSES}
FOOD_CLASS_MAP={k:{'nome':a[0],'taco_id':a[1],'grams':a[2],'note':a[3]} for k,a in CLASSES.items()}
BRAZIL_LABELS={
 'rice':1,'beans':2,'chicken':3,'beef':4,'salad':5,'fries':6,'egg':7,'noodles':8,
 'tomato':10,'carrot':11,'pumpkin':12,'milanesesteak':13,'farofa':14,'beet':15,
 'mashed':16,'pastaredsauce':17,'cabbage':18,'corn':19,'mayo(potato salad)':20,'cassava':21,
}
MYFOOD_LABELS={2:9,3:2,4:7,5:3,6:1,7:5,8:8,9:7,10:4}
MYFOOD_NAME_LABELS={'apple':9,'bean':2,'boiled_egg':7,'chicken_breast':3,'rice':1,'salad':5,'spaghetti':8,'fried_egg':7,'steak':4}
