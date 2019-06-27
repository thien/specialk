## Transformer

`load()` loads an `encoder()` and `decoder()` components of the model. the `decoder()` component also contains `target_word_projection()`.

`save()` saves the encoder and decoder parts seperately.
Both the encoder and the decoder contain model parameters and other
metadata.

They're seperated in order to swap encoders and decoders arbitarily, a necessary component for style transfer.