import numpy as np
import hashlib
import colorsys
import cv2 as cv
from abc import ABC, abstractmethod
from boxmot.utils import logger as LOGGER
from boxmot.utils.iou import AssociationFunction

# 假设 TrackState 类已经定义
# class TrackState:
#     New = 0
#     Lost = 1
#     Removed = 2
#

class RgbtBaseTracker(ABC):
    def __init__(
            self,
            det_thresh: float = 0.3,
            max_age: int = 30,
            min_hits: int = 3,
            iou_threshold: float = 0.3,
            max_obs: int = 50,
            nr_classes: int = 80,
            per_class: bool = False,
            asso_func: str = 'iou'
    ):
        """
        初始化 NewBaseTracker 对象，使用检测阈值、最大年龄、最小命中数、
        以及交并比（IOU）阈值来跟踪视频帧中的对象。

        参数:
        - det_thresh (float): 考虑检测的检测阈值。
        - max_age (int): 跟踪在被认为丢失之前的最大年龄。
        - min_hits (int): 跟踪被认为确认之前的最小检测命中数。
        - iou_threshold (float): 确定检测与跟踪之间匹配的 IOU 阈值。

        属性:
        - frame_count (int): 处理的帧数计数器。
        - active_tracks (list): 用于保存活动跟踪的列表，在子类中可能有不同的使用方式。
        """
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.max_obs = max_obs
        self.min_hits = min_hits
        self.per_class = per_class  # 是否按类别跟踪
        self.nr_classes = nr_classes
        self.iou_threshold = iou_threshold
        self.last_emb_size = None
        self.asso_func_name = asso_func

        self.frame_count = 0
        self.active_tracks = []
        self.per_class_active_tracks = None
        self._first_frame_processed = False  # 标记第一帧是否已处理

        # 初始化按类别活动跟踪
        if self.per_class:
            self.per_class_active_tracks = {}
            for i in range(self.nr_classes):
                self.per_class_active_tracks[i] = []

        if self.max_age >= self.max_obs:
            LOGGER.warning("Max age > max observations, increasing size of max observations...")
            self.max_obs = self.max_age + 5
            print("self.max_obs", self.max_obs)

    @abstractmethod
    def update(self, visible_dets: np.ndarray, infrared_dets: np.ndarray, visible_img: np.ndarray, infrared_img: np.ndarray,
               visible_embs: np.ndarray = None, infrared_embs: np.ndarray = None) -> np.ndarray:
        """
        抽象方法，用于使用新的检测结果更新跟踪器以处理新的帧。此方法
        应由子类实现。

        参数:
        - visible_dets,infrared_dets  (np.ndarray): 当前帧的检测数组。
        - visible_img (np.ndarray): 当前帧的可见光图像数组。
        - infrared_img (np.ndarray): 当前帧的红外图像数组。
        - embs (np.ndarray, optional): 与检测相关的嵌入，如果有的话。

        抛出:
        - NotImplementedError: 如果子类未实现此方法。
        """
        raise NotImplementedError("The update method needs to be implemented by the subclass.")

    def get_class_dets_n_embs(self, dets, embs, cls_id):
        # 初始化空数组用于检测和嵌入
        class_dets = np.empty((0, 6))
        class_embs = np.empty((0, self.last_emb_size)) if self.last_emb_size is not None else None

        # 检查是否有检测结果
        if dets.size > 0:
            class_indices = np.where(dets[:, 5] == cls_id)[0]
            class_dets = dets[class_indices]

            if embs is not None:
                # 断言如果提供了嵌入，则它们的元素数量应与检测结果相同
                assert dets.shape[0] == embs.shape[
                    0], "Detections and embeddings must have the same number of elements when both are provided"

                if embs.size > 0:
                    class_embs = embs[class_indices]
                    self.last_emb_size = class_embs.shape[1]  # 更新最后已知的嵌入大小
                else:
                    class_embs = None
        return class_dets, class_embs

    @staticmethod
    def on_first_frame_setup(method):
        """
        装饰器，仅在第一帧时执行设置。
        这确保初始化任务（如设置关联函数）仅在第一帧执行一次，后续帧跳过。
        """

        def wrapper(self, *args, **kwargs):
            # 如果设置尚未完成，则执行设置
            if not self._first_frame_processed:
                visible_img = args[1]
                self.h, self.w = visible_img.shape[0:2]
                self.asso_func = AssociationFunction(w=self.w, h=self.h, asso_mode=self.asso_func_name).asso_func

                # 标记第一帧设置已完成
                self._first_frame_processed = True

            # 调用原始方法（如 update）
            return method(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def per_class_decorator(update_method):
        """
        用于 update 方法的装饰器，以处理按类别处理。
        """

        def wrapper(self, visible_dets: np.ndarray, visible_img: np.ndarray, infrared_dets: np.ndarray,  infrared_img: np.ndarray,
               visible_embs: np.ndarray = None, infrared_embs: np.ndarray = None):

            # 处理不同类型的输入
            if visible_dets is None or len(visible_dets) == 0:
                dets = np.empty((0, 6))

            if self.per_class:
                # 初始化一个数组来存储每个类别的跟踪结果
                visible_per_class_tracks = []
                infrared_per_class_tracks = []

                # 所有类别的帧数相同
                frame_count = self.frame_count

                for cls_id in range(self.nr_classes):
                    # 获取当前类别的检测和嵌入
                    class_dets, class_embs = self.get_class_dets_n_embs(dets, embs, cls_id)

                    LOGGER.debug(
                        f"Processing class {int(cls_id)}: {class_dets.shape} with embeddings {class_embs.shape if class_embs is not None else None}")

                    # 激活当前类别的特定活动跟踪
                    self.active_tracks = self.per_class_active_tracks[cls_id]
                    # 重置每个类别的帧数
                    self.frame_count = frame_count

                    # 使用装饰后的方法更新检测结果
                    visible_tracks, infrared_tracks = update_method(self, dets=class_dets, visible_img=visible_img, infrared_img=infrared_img,
                                           embs=class_embs)

                    # 保存更新后的活动跟踪
                    self.per_class_active_tracks[cls_id] = self.active_tracks

                    if visible_tracks.size > 0:
                        visible_per_class_tracks.append(visible_tracks)
                    if infrared_tracks.size > 0:
                        infrared_per_class_tracks.append(infrared_tracks)

                # 帧数加 1
                self.frame_count = frame_count + 1

                visible_per_class_tracks = np.vstack(
                    visible_per_class_tracks) if visible_per_class_tracks else np.empty((0, 8))
                infrared_per_class_tracks = np.vstack(
                    infrared_per_class_tracks) if infrared_per_class_tracks else np.empty((0, 8))

                return np.vstack(visible_per_class_tracks), np.vstack(infrared_per_class_tracks)

            else:
                # 如果 per_class 为 False，则一次性处理所有检测结果
                return update_method(self, dets=dets, visible_img=visible_img, infrared_img=infrared_img, embs=embs)

        return wrapper

    def check_inputs(self, dets, visible_img):
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            visible_img, np.ndarray
        ), f"Unsupported 'visible_img' input format '{type(visible_img)}', valid format is np.ndarray"
        assert (
                len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
                dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

    def id_to_color(self, id: int, saturation: float = 0.75, value: float = 0.95) -> tuple:
        """
        为给定的 ID 生成一致的唯一 BGR 颜色，使用哈希。

        参数:
        - id (int): 用于生成颜色的唯一标识符。
        - saturation (float): HSV 空间中的饱和度值。
        - value (float): HSV 空间中的值（亮度）。

        返回:
        - tuple: 表示 BGR 颜色的元组。
        """

        # 对 ID 进行哈希以获得一致的唯一值
        hash_object = hashlib.sha256(str(id).encode())
        hash_digest = hash_object.hexdigest()

        # 将哈希的前几个字符转换为整数，并将其映射到 0 到 1 之间的值作为色调
        hue = int(hash_digest[:8], 16) / 0xffffffff

        # 将 HSV 转换为 RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)

        # 将 RGB 从 0-1 范围转换为 0-255 范围并格式化为十六进制
        rgb_255 = tuple(int(component * 255) for component in rgb)
        hex_color = '#%02x%02x%02x' % rgb_255
        # 去掉 '#' 字符并将字符串转换为 RGB 整数
        rgb = tuple(int(hex_color.strip('#')[i:i + 2], 16) for i in (0, 2, 4))

        # 将 RGB 转换为 BGR 以用于 OpenCV
        bgr = rgb[::-1]

        return bgr

    def plot_box_on_img(self, img: np.ndarray, box: tuple, conf: float, cls: int, id: int, thickness: int = 2,
                        fontscale: float = 0.5) -> np.ndarray:
        """
        在图像上绘制带有 ID、置信度和类别信息的边界框。

        参数:
        - img (np.ndarray): 要绘制的图像数组。
        - box (tuple): 边界框坐标，格式为 (x1, y1, x2, y2)。
        - conf (float): 检测的置信度得分。
        - cls (int): 检测的类别 ID。
        - id (int): 检测的唯一标识符。
        - thickness (int): 边界框的厚度。
        - fontscale (float): 文本的字体缩放比例。

        返回:
        - np.ndarray: 绘制了边界框的图像数组。
        """

        img = cv.rectangle(
            img,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            self.id_to_color(id),
            thickness
        )
        img = cv.putText(
            img,
            f'id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}',
            (int(box[0]), int(box[1]) - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            fontscale,
            self.id_to_color(id),
            thickness
        )
        return img

    def plot_trackers_trajectories(self, img: np.ndarray, observations: list, id: int) -> np.ndarray:
        """
        根据历史观测绘制跟踪对象的轨迹。轨迹中的每个点
        用一个圆表示，最近的观测点的厚度增加，以可视化运动路径。

        参数:
        - img (np.ndarray): 要绘制轨迹的图像数组。
        - observations (list): 跟踪对象的历史观测边界框坐标列表，每个观测格式为 (x1, y1, x2, y2)。
        - id (int): 跟踪对象的唯一标识符，用于可视化的颜色一致性。

        返回:
        - np.ndarray: 绘制了轨迹的图像数组。
        """
        for i, box in enumerate(observations):
            trajectory_thickness = int(np.sqrt(float(i + 1)) * 1.2)
            img = cv.circle(
                img,
                (int((box[0] + box[2]) / 2),
                 int((box[1] + box[3]) / 2)),
                2,
                color=self.id_to_color(int(id)),
                thickness=trajectory_thickness
            )
        return img

    def plot_results(self, img: np.ndarray, show_trajectories: bool, thickness: int = 2,
                     fontscale: float = 0.5) -> np.ndarray:
        """
        在图像上可视化所有活动跟踪的轨迹。对于每个跟踪，
        它绘制最新的边界框以及运动路径（如果观测历史长度大于 2）。
        这有助于理解每个跟踪对象的运动模式。

        参数:
        - img (np.ndarray): 要绘制轨迹和边界框的图像数组。
        - show_trajectories (bool): 是否显示轨迹。
        - thickness (int): 边界框的厚度。
        - fontscale (float): 文本的字体缩放比例。

        返回:
        - np.ndarray: 绘制了轨迹和所有活动跟踪的边界框的图像数组。
        """

        # 如果字典中有值
        if self.per_class_active_tracks is not None:
            for k in self.per_class_active_tracks.keys():
                active_tracks = self.per_class_active_tracks[k]
                for a in active_tracks:
                    if a.history_observations:
                        if len(a.history_observations) > 2:
                            box = a.history_observations[-1]
                            img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                            if show_trajectories:
                                img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
        else:
            for a in self.active_tracks:
                if a.history_observations:
                    if len(a.history_observations) > 2:
                        box = a.history_observations[-1]
                        img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                        if show_trajectories:
                            img = self.plot_trackers_trajectories(img, a.history_observations, a.id)

        return img