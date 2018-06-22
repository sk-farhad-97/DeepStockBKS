from keras.models import model_from_json
import os


def save_model(model, file_path, name):
    # serialize model to JSON
    try:
        full_dir_path = os.path.join(file_path, name + ".json")
        model_json = model.to_json()
        with open(full_dir_path, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        full_dir_path = os.path.join(file_path, name + ".h5")
        model.save_weights(full_dir_path)
        return False
    except:
        print('Saving model failed!!')
        return True


def load_model(file_path, name):
    try:
        full_dir_path = os.path.join(file_path, name + ".json")
        json_file = open(full_dir_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        full_dir_path = os.path.join(file_path, name + ".h5")
        loaded_model.load_weights(full_dir_path)
        return loaded_model
    except:
        print('Get model failed!!')
        return False
