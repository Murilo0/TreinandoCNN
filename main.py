import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# 1. Carregar os dados do CSV
data = pd.read_csv("digitosmanuscritos.csv")

# Separar características (pixels) e rótulos (dígitos)
X = data.iloc[:, 1:].values.astype('float32')
y = data.iloc[:, 0].values

# Normalizar os pixels (0-1)
X /= 255.0

# Redimensionar para o formato (n_amostras, 28, 28, 1)
X = X.reshape(-1, 28, 28, 1)

# One-hot encoding dos rótulos
y = tf.keras.utils.to_categorical(y, 10)

# 2. Definir a CNN
def criar_modelo():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# 3. Função para treinar, avaliar, plotar gráficos e salvar matriz de confusão
def treinar_e_testar(X, y, proporcao_treino):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=proporcao_treino, random_state=42, stratify=y.argmax(axis=1)
    )

    model = criar_modelo()

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=128,
        verbose=1,
        validation_split=0.1,
        callbacks=[early_stop]
    )

    # Plotar gráficos de perda e acurácia
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Perda durante o treino')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia durante o treino')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()

    filename_plot = f"training_plot_{round(proporcao_treino * 100)}_{round((1 - proporcao_treino) * 100)}.png"
    plt.savefig(filename_plot)
    plt.close()
    print(f"Gráficos de treino salvos em: {filename_plot}")

    # Avaliação
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    divisao = f"{round(proporcao_treino * 100)}_{round((1 - proporcao_treino) * 100)}"
    print(f"Acurácia com {divisao.replace('_', '/')}: {acc:.4f}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(classification_report(y_true, y_pred_classes))
    cm = confusion_matrix(y_true, y_pred_classes)

    # Criar a matriz em markdown
    markdown_lines = []
    header = "| |" + "|".join(str(i) for i in range(10)) + "|\n"
    separator = "|---" * 10 + "|---|\n"
    markdown_lines.append(header)
    markdown_lines.append(separator)
    for i, row in enumerate(cm):
        row_str = f"|{i}|" + "|".join(str(cell) for cell in row) + "|\n"
        markdown_lines.append(row_str)

    filename = f"confusion_matrix_{divisao}.md"
    with open(filename, "w") as f:
        f.write(f"# Confusion Matrix ({divisao.replace('_', '/')})\n\n")
        f.write(f"**Accuracy:** {acc:.4f}\n\n")
        f.writelines(markdown_lines)

    print(f"\nMatriz de confusão salva como: {filename}")

# 4. Realizar os testes com diferentes proporções
proporcoes = [0.8, 0.9, 0.7]
for prop in proporcoes:
    print("\n" + "=" * 50)
    treinar_e_testar(X, y, prop)