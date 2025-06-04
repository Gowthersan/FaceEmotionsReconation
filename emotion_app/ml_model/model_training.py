import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models


# Générateur de données pour charger par lots
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=64):
        self.df = df
        self.batch_size = batch_size
        self.indices = np.arange(len(df))
        
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        
        X = []
        y = []
        for _, row in batch_df.iterrows():
            # Conversion des pixels
            pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.float32)
            pixels = pixels.reshape(48, 48, 1) / 255.0
            X.append(pixels)
            y.append(int(row['emotion']))  # Conversion explicite en int
            
        # Conversion des labels en array numpy avant de les catégoriser
        y_array = np.array(y)
        y_categorical = tf.keras.utils.to_categorical(y_array, num_classes=7)
        return np.array(X), y_categorical

# Chargement du dataset FER2013 avec chemin absolu
def load_data():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'fer2013.csv')
    df = pd.read_csv(csv_path)
    return df

# Architecture CNN simplifiée
def create_model():
    model = models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=(48, 48, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.2),
        
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.4),
        
        layers.Dense(7, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Entraînement avec gestion d'erreurs
def train_and_save_model():
    try:
        print("Chargement des données...")
        df = load_data()
        
        # Division des données
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Création des générateurs
        train_generator = DataGenerator(train_df, batch_size=64)
        test_generator = DataGenerator(test_df, batch_size=64)
        
        print("Création du modèle...")
        model = create_model()
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        
        print("Début de l'entraînement...")
        history = model.fit(
            train_generator,
            epochs=50,
            validation_data=test_generator,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Sauvegarde du modèle
        model.save('fer_model.h5')
        print("Modèle entraîné et sauvegardé avec succès!")
        
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {e}")
        raise

if __name__ == "__main__":
    train_and_save_model()
