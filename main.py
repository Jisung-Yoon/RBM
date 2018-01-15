from util import *
from ops import *
from model import *

SEED = 111  # seed for shuffle batch
BATCH_SIZE = 20
TRAINING_EPOCHS = 1
HIDDEN_SIZE = 100
ALGORITHM_TYPE = 'PCD'
NAME = 'RBM_PCD'


def main():
    images, _ = load_mnist_datasets(SEED)
    model = RBM(hidden_size=HIDDEN_SIZE, algorithm_type=ALGORITHM_TYPE, name=NAME)

    total_batch = int(len(images) / BATCH_SIZE)
    for epoch in range(TRAINING_EPOCHS):
        total_error = 0
        check = True  # check variable for PCD

        for idx in range(total_batch):
            batch_images = images[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            total_error += model.train(batch_images, generating=check) / total_batch
            if check:
                check = False
        # print('Epoch: ' + str((epoch + 1)) + ': ' + str(total_error))

        if epoch % 50 == 0:
            generated_images = model.generate_visible()
            reshaped_and_save_images(generated_images, model.result_path, epoch)
    print('Learning Finished!')

    hidden_array = model.calculate_hidden(images)
    print(calculate_entropy(hidden_array))

if __name__ == '__main__':
    main()