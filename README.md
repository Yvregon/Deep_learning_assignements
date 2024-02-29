# Projet Initiation aux réseaux de neurones avec PyTorch


## Contenu du Projet
Le projet est organisé de la manière suivante : 

Dans le dossier **Intro**, on trouve un classifieur d'images 28 par 28 pixels, en catégorie d'habillement, entraîné et testé sur la base *fashion_MNIST*.

Dans le dossier **Segmentation** on trouve un réseau UNet qui fait de la segmentation d'images en 14 label différents, dont un label "unknown". Ce réseau est entrainé sur la base de données *Stanford 2D-3D S dataset*


## Installation 
Pour cloner le projet faire :

```bash
git clone https://github.com/Yvregon/Deep_learning_assignements.git
```


Après vous êtes mis dans l'environnement python de votre choix, les modules à installer peut être installés grâce à : 

```bash
pip install -r requirements.txt
```

## Utilisation
Pour lancer le TP 1 : 

```bash
cd Intro 

./launch.sh
```

Pour lancer le TP2.b : 
```bash
./TP2.sh
```