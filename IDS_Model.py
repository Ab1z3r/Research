import tensorflow as tf
class IDS_Model:

    def __init__(self):
        self.num_classes = 6
        self.ids_model = self.load_ids()

    def load_ids(self):
        print("[*] Loading IDS_Model...")
        model = tf.keras.models.load_model('./IDS_Model.h5')
        print("[+] Finished loading BackBox IDS model...")
        return model

    def IDS_loss(self, predictions, target_labels):
        target_one_hot = tf.one_hot(target_labels, depth=6)
        preds_one_hot = tf.one_hot(predictions, depth=6)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=target_one_hot, logits=preds_one_hot)
        loss = tf.reduce_mean(cross_entropy)
        return loss
