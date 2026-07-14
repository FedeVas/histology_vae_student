# Histology VAE

Учебно-исследовательский проект по обучению Variational Autoencoder на H&E histology patches.

Цель проекта — построить воспроизводимый pipeline для unsupervised representation learning на гистологических изображениях с дальнейшим анализом reconstruction, latent space, clustering и patient-level representations.

## Текущий этап

На данный момент реализованы:

* конфигурация проекта через YAML;
* автоматический выбор CPU, CUDA или MPS;
* воспроизводимая установка random seed;
* проверка forward pass, backward pass и optimizer step;
* patient-level разделение данных без утечки пациентов;
* PyTorch Dataset и DataLoader;
* train и evaluation transforms;
* генератор синтетических histology-like patches;
* unit tests для вычислительного устройства и data pipeline.

Синтетические изображения используются только для инженерной проверки pipeline и не предназначены для биологических выводов.

## Структура проекта

```text
histology-vae/
├── configs/
├── data/
│   ├── metadata/
│   ├── patches/
│   └── raw/
├── notebooks/
├── outputs/
├── reports/
├── src/
│   ├── analysis/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   └── utils/
├── tests/
├── requirements.txt
└── README.md
```

## Установка

Создание виртуального окружения на Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Установка CPU-версии PyTorch:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Установка остальных зависимостей:

```powershell
python -m pip install -r requirements.txt
```

## Проверка окружения

```powershell
python -m src.check_environment
```

Скрипт проверяет:

* загрузку конфигурации;
* выбор вычислительного устройства;
* создание тензоров;
* forward pass;
* backward pass;
* optimizer step.

## Проверка data pipeline

```powershell
python -m src.check_data_pipeline
```

При первом запуске будет создан небольшой синтетический dataset для проверки metadata, patient-level splits, Dataset и DataLoader.

## Запуск тестов

```powershell
python -m pytest -q
```

## Вычислительные устройства

В `configs/vae_base.yaml` можно выбрать:

```yaml
device:
  accelerator: auto
```

Поддерживаемые значения:

* `auto`;
* `cpu`;
* `cuda`;
* `mps`.

В режиме `auto` используется следующий приоритет:

```text
CUDA → MPS → CPU
```

## Следующие этапы

Планируется реализовать:

1. convolutional encoder и decoder;
2. VAE reparameterization;
3. reconstruction и KL losses;
4. training и validation loops;
5. reconstruction metrics;
6. latent-space extraction;
7. PCA, UMAP и clustering;
8. patient-level aggregation;
9. сравнение с pretrained image encoders.

## Ограничения

Реальные histology datasets, whole-slide images, извлечённые patches, checkpoints и результаты экспериментов не хранятся в Git-репозитории.

Для больших данных в дальнейшем планируется использовать внешнее хранилище или систему версионирования данных.
