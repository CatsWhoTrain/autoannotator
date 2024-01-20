# Human Detection

The version 0.2.0 has introduced several state-of-the-art models designed specifically for detecting humans' full bodies accurately and efficiently. We have made it a priority to incorporate the most advanced techniques in this field.

The following models were included:
- **InternImage** from [InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions](https://arxiv.org/abs/2211.05778)
- **IterDETR** from [Progressive End-to-End Object Detection in Crowded Scenes](https://arxiv.org/abs/2203.07669)
- **RT-DETR** from [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
- **UniHCP** from [UniHCP: A Unified Model for Human-Centric Perceptions](https://arxiv.org/abs/2303.02936)

### Tests

We have conducted tests to evaluate how utilizing an ensemble of these models enhances human detection training performance. In our use case, we employed the [CUHK-SYSU](https://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html) dataset which was automatically labeled by the aforementioned models without any manual intervention. Subsequently, we trained an RT-DETR model with ResNet-18 backbone (640x640 input resolution) on both the original and auto-annotated versions of CUHK-SYSU dataset. Finally, we assessed the best checkpoints using test sets of [CrowdHuman](https://www.crowdhuman.org/) and [WiderPerson](http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/) datasets for testing purposes.

We have utilized AP50 as a metric for evaluating our trained model's performance. On the CrowdHuman dataset, we observed an improvement in AP50 from 52.30% to 77.31%. Similarly, on the WiderPerson dataset, there was also a progression with AP50 increasing from 54.66% to 63.71%.

### Difficulties
During the ONNX conversion process we encountered several issues:
1. We have trained DDQ-DETR with more advanced backbone by ourselves and were able to achieve top-2 on Object Detection on CrowdHuman (AP50 = 95.33%) but were not able yet to convert the model to ONNX because it was trained using mmdetection framework, and we figured out that DDQ is not yet supported by mmdeploy which converts mmdetection-based models.
2. The InternImage model requires an external implementation of custom operations which are currently only available on Linux due to author-specific implementations. This means that if one wishes to use this model, they must utilize a Linux operating system.
3. We were unable to publish the ONNX model for UniHCP because the original authors distribute weights individually upon request rather than publicly sharing them. Once we have sufficient hardware resources available, our intention is to train the model ourselves so that it can be utilized by all users with fewer restrictions.
4. The InternImage-XL version included in this conversion was trained on a COCO Object Detection and Instance Segmentation dataset instead of one specifically tailored for human detection tasks. We have requested the authors to share their weights fone-tuned on CrowdHuman accordingly.