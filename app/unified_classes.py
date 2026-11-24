"""Rótulos visuais comuns aos datasets MyFood e Brazilian Food Images.

IDs visuais são distintos dos códigos TACO: a TACO descreve composição nutricional,
não anotações de imagens. Cada correspondência nutricional é uma HIPÓTESE editável.
"""
CLASSES={
 1:('Arroz',3,125,''),  # TACO: arroz tipo 1 cozido; na tela só 'Arroz' (NOMES_EXIBIDOS)
 2:('Feijão',561,110,''),  # linha da TACO escolhida pela cor da região: ver VARIANTES
 3:('Frango',410,130,''),  # TACO: peito sem pele grelhado; na tela só 'Frango'
 4:('Carne bovina',377,130,''),  # TACO: patinho sem gordura grelhado; na tela só 'Carne bovina'
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
 16:('Purê',91,115,''),  # sem item na TACO: um item só, calculado pela receita (ver RECEITAS_UM_ITEM)
 17:('Massa com molho vermelho',37,140,'Massa fresca cozida como PROXY (a TACO não tem massa ao sugo): acrescente "Tomate, molho industrializado" (#159) para o molho'),
 18:('Repolho',149,55,'Repolho branco cru como aproximação'),
 19:('Milho',45,70,'Milho verde enlatado e drenado como aproximação'),
 20:('Salada com maionese',545,90,'Salada de legumes com maionese como aproximação'),
 21:('Mandioca',129,100,'Mandioca cozida como aproximação'),
 # 22 a 28: fotos do MyFoodRepo (ver MYFOODREPO_LABELS). Modelos de 21 classes simplesmente não as detectam.
 22:('Brócolis',100,60,'Brócolis cozido como aproximação'),
 23:('Batata cozida',91,120,'Batata inglesa cozida'),
 24:('Peixe',301,120,'Filé de merluza assado como aproximação: confirme o peixe e o preparo'),
 25:('Pão',53,50,'Pão francês como aproximação: confirme o tipo de pão'),
 26:('Cebola',107,30,'Cebola crua como aproximação'),
 27:('Hambúrguer (sanduíche)',417,150,'Sem item na TACO: lançado como pão + hambúrguer bovino grelhado (ver RECIPES)'),
 28:('Lasanha',37,250,'Sem item na TACO: lançada como massa + carne moída + molho de tomate + muçarela (ver RECIPES)'),
}
N_CLASS=len(CLASSES)
CLASS_LABELS={k:v[0] for k,v in CLASSES.items()}
# Mais de 20 cores: distinção visual, não graus de confiança.
PALETTE=['#70b8ee','#8f6e50','#edaa6b','#b96564','#6fbc92','#e5b84c','#ecce79','#bb90d5','#f27991','#e86d69','#ed8f4a','#e3a05b','#ad7760','#cfb97c','#c76992','#eac69a','#ce5a5a','#71b36d','#e9d353','#c8bda5','#a28c64',
         '#3f9e5a','#d8c48f','#7fa7c9','#c9955c','#b58fc4','#9c5b3c','#d9734e']
LABEL_COLORS={k:PALETTE[k-1] for k in CLASSES}
FOOD_CLASS_MAP={k:{'nome':a[0],'taco_id':a[1],'grams':a[2],'note':a[3]} for k,a in CLASSES.items()}
# A rede só sabe que é feijão; o tipo sai da cor da região na foto. Mais da metade dos pixels
# escuros -> feijão preto (#567); senão carioca (#561, o mais consumido). Na tela aparece só o nome genérico.
VARIANTES={2:{'nome_exibido':'Feijão cozido','escuro':567,'claro':561}}
# Nome curto na tela no lugar do nome completo da TACO (a linha da TACO continua a mesma).
NOMES_EXIBIDOS={1:'Arroz',3:'Frango',4:'Carne bovina'}
# Preparos que a TACO não tem como item único: a classe vira vários itens da TACO.
# Frações em massa de uma receita caseira típica, editáveis pelo usuário.
# Purê: 1 kg de batata cozida + 200 g de leite integral + 50 g de manteiga (~1250 g) -> 80/16/4.
# O leite integral líquido da TACO (#458) não tem energia nem macronutrientes, então os 16% de leite
# entram como leite integral em pó (#459, 2%) + água (14%; taco_id None = água, zero de tudo).
RECIPES={
 16:(('batata',91,0.80),('leite integral em pó',459,0.02),('água',None,0.14),('manteiga com sal',261,0.04)),
 # Sanduíche: pão (~60 g; a TACO não tem pão de hambúrguer, usa o francês) + hambúrguer (~90 g) -> 40/60.
 27:(('pão',53,0.40),('hambúrguer grelhado',417,0.60)),
 # Lasanha à bolonhesa caseira, frações em massa do prato montado e assado.
 28:(('massa',37,0.40),('carne moída',326,0.25),('molho de tomate',159,0.20),('muçarela',463,0.15)),
}
# Receitas mostradas como UM item só (nome curto na tela). Os nutrientes por 100 g são a média
# ponderada dos ingredientes de RECIPES; o item ganha um id fora da faixa da TACO (9000 + classe).
RECEITAS_UM_ITEM={16:'Purê'}
RECEITA_ID_BASE=9000
BRAZIL_LABELS={
 'rice':1,'beans':2,'chicken':3,'beef':4,'salad':5,'fries':6,'egg':7,'noodles':8,
 'tomato':10,'carrot':11,'pumpkin':12,'milanesesteak':13,'farofa':14,'beet':15,
 'mashed':16,'pastaredsauce':17,'cabbage':18,'corn':19,'mayo(potato salad)':20,'cassava':21,
}
MYFOOD_LABELS={2:9,3:2,4:7,5:3,6:1,7:5,8:8,9:7,10:4}
MYFOOD_NAME_LABELS={'apple':9,'bean':2,'boiled_egg':7,'chicken_breast':3,'rice':1,'salad':5,'spaghetti':8,'fried_egg':7,'steak':4}
# MyFoodRepo v2.0 (AIcrowd Food Recognition 2022): nome da categoria COCO -> classe comum.
# Categorias fora deste mapa viram "outro alimento" (ignoradas no treino), como o código 1 do MyFood.
# Decisões: coxa de frango entra em Frango (só 85 fotos); 'hamburger' é só a carne (-> Carne bovina)
# e 'hamburger-bun' é o pão; o sanduíche inteiro é 'hamburger-bread-meat-ketchup'.
MYFOODREPO_LABELS={
 'rice':1,'rice-basmati':1,
 'chicken':3,'chicken-breast':3,'chicken-leg':3,'chicken-cut-into-stripes-only-meat':3,
 'beef':4,'beef-filet':4,'beef-sirloin-steak':4,'beef-roast':4,'beef-cut-into-stripes-only-meat':4,'hamburger':4,
 'salad-leaf-salad-green':5,'mixed-salad-chopped-without-sauce':5,'salad-lambs-ear':5,'salad-rocket':5,
 'chips-french-fries':6,'country-fries':6,
 'egg':7,
 'pasta':8,'pasta-spaghetti':8,'pasta-penne':8,'pasta-twist':8,'pasta-noodles':8,'pasta-linguini-parpadelle-tagliatelle':8,
 'pasta-hornli':8,'pasta-wholemeal':8,'pasta-in-butterfly-form-farfalle':8,'pasta-in-conch-form':8,
 'apple':9,
 'tomato-raw':10,
 'carrot-raw':11,'carrot-steamed-without-addition-of-salt':11,
 'pumpkin':12,
 'beetroot-steamed-without-addition-of-salt':15,'beetroot-raw':15,
 'mashed-potatoes-prepared-with-full-fat-milk-with-butter':16,
 'white-cabbage':18,'red-cabbage':18,
 'corn':19,'sweet-corn-canned':19,
 'potato-salad-with-mayonnaise-yogurt-dressing':20,
 'broccoli':22,
 'potatoes-steamed':23,
 'fish':24,'salmon':24,
 'bread':25,'bread-white':25,'bread-wholemeal':25,'bread-whole-wheat':25,'bread-grain':25,'bread-sourdough':25,
 'bread-french-white-flour':25,'bread-half-white':25,'bread-toast':25,'bread-wholemeal-toast':25,'bread-black':25,
 'bread-rye':25,'bread-spelt':25,'bread-5-grain':25,'bread-nut':25,'bread-olive':25,'bread-ticino':25,
 'gluten-free-bread':25,'roll-of-half-white-or-white-flour-with-large-void':25,'hamburger-bun':25,
 'onion':26,
 'hamburger-bread-meat-ketchup':27,
 'lasagne-meat-prepared':28,'lasagne-vegetable-prepared':28,
}
NEW_CLASSES=tuple(range(22,N_CLASS+1))  # classes que só existem graças ao MyFoodRepo
