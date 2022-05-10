


i = 50
for x, file, y in zip(x_train, combined_list, y_train):
    i += 1
    plt.imsave(f"results/grad_cam/original{i}.png", x.reshape((1444, 1448)), cmap="gray")
    with tf.GradientTape() as tape:
        last_conv_layer = eval_model.get_layer('conv2d_59')
        iterate = tf.keras.models.Model([eval_model.inputs], [eval_model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x.reshape(1, 1444, 1448, 1))
        class_out = model_out[:, tf.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = k.backend.mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    print(f"\nFrom File {file}:")
    print("Predicted:", model_out[0], "True: ", y)
    plt.imsave(f"results/grad_cam/grad_cam{i}.png", np.maximum(heatmap.numpy().reshape(
        last_conv_layer.shape[1:3].as_list()), 0)
               )
    if i == 50:
        break