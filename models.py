import segmentation_models_pytorch as smp


class ModelFactory:
    @staticmethod
    def create_model(model_name):
        models = {
            'unet': smp.Unet(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=3,
                classes=1
            ),
            'unet_boundary': smp.Unet(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=3,
                classes=1,
                decoder_attention_type='scse'
            ),
            'attention_unet': smp.Unet(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=3,
                classes=1,
                decoder_attention_type='scse'
            ),
            'unet_plusplus': smp.UnetPlusPlus(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=3,
                classes=1
            ),
            'deeplabv3': smp.DeepLabV3Plus(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=3,
                classes=1
            ),
            'pspnet': smp.PSPNet(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=3,
                classes=1
            ),
            'fpn': smp.FPN(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=3,
                classes=1
            ),
            'pan': smp.PAN(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=3,
                classes=1
            ),
            'linknet': smp.Linknet(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=3,
                classes=1
            ),
            'manet': smp.MAnet(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=3,
                classes=1
            )
        }

        return models[model_name]