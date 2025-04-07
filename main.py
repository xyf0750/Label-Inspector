import tkinter as tk
import os
import sys
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import torch
import numpy as np
import threading
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes

class YOLOInspector:
    def __init__(self, master):
        #依赖检查
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")

        self.master = master
        master.title("YOLOv5检测系统")
        
        # 初始化配置
        self.model = None
        self.available_models = {
            "标签定位模型": "runs/train/exp17/weights/best.pt",
            "缺陷检测模型": "runs/train/defect_finetune-200-v27/weights/best.pt"
        }
        
        # 初始化界面
        self.create_widgets()
        self.setup_layout()
        self.load_default_model()
        self._init_torch_security()
        self.load_default_model()
        # 使用系统级路径解析
        self.base_dir = self._get_base_path()
        self.available_models = {
            "标签定位模型": self.resource_path('runs/train/exp17/weights/best.pt'),
            "缺陷检测模型": self.resource_path('runs/train/defect_finetune-200-v27/weights/best.pt')
        }
        # 初始化日志系统
        self._init_logging()
    
    def _init_logging(self):
        """重定向标准输出到日志面板"""
        class LogRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
                
            def write(self, message):
                if self.text_widget:
                    self.text_widget.configure(state='normal')
                    self.text_widget.insert('end', message)
                    self.text_widget.see('end')
                    self.text_widget.configure(state='disabled')
                
            def flush(self):
                pass

        # 确保日志文本框已初始化
        if hasattr(self, 'log_text'):
            # 重定向stdout/stderr
            sys.stdout = LogRedirector(self.log_text)
            sys.stderr = LogRedirector(self.log_text)
        
        # 显示启动信息
        print(f"🟢 系统启动 | PyTorch {torch.__version__} | CUDA {'可用' if torch.cuda.is_available() else '不可用'}")
    
    def _get_base_path(self):
        """动态获取资源根路径"""
        if getattr(sys, 'frozen', False):
            # 打包后路径指向临时解压目录
            return sys._MEIPASS
        else:
            # 开发环境使用项目根目录
            return os.path.dirname(os.path.abspath(__file__))

    def resource_path(self, relative_path):
        """将相对路径转换为绝对路径"""
        return os.path.join(self.base_dir, relative_path.replace('/', os.sep))

    def _init_torch_security(self):
        """解决PyTorch 2.6+的安全加载限制"""
        from torch.serialization import add_safe_globals
        try:
            # 导入YOLO自定义类并添加到安全列表
            from models.yolo import DetectionModel, ClassificationModel
            from models.common import Conv, Bottleneck
            add_safe_globals([DetectionModel, ClassificationModel, Conv, Bottleneck])
        except ImportError as e:
            print(f"⚠️ 安全类注册失败: {str(e)}")
        
    def load_default_model(self):
        """加载默认模型"""
        model_name = self.model_selector.get()
        model_path = self.available_models[model_name]
        self.load_model(model_path)
        self.lbl_model.config(text=f"当前模型: {model_name}")

    def load_model(self, model_path):
        """安全加载模型"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 使用attempt_load加载模型
            self.model = attempt_load(model_path, device=device).eval()
            
            print(f"✅ 模型加载成功 | 设备: {device} | 路径: {model_path}")
            self.update_status(f"模型加载成功: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            self.model = None
            self.update_status(f"模型加载失败: {os.path.basename(model_path)}")
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    

    def create_widgets(self):
        """创建界面组件"""
        # 工具栏
        self.toolbar = ttk.Frame(self.master)
        
        # 模型选择
        self.model_selector = ttk.Combobox(
            self.toolbar, 
            values=list(self.available_models.keys()),
            state="readonly"
        )
        self.model_selector.current(0)
        self.model_selector.bind("<<ComboboxSelected>>", self.change_model)

        # 日志面板
        self.log_panel = ttk.LabelFrame(self.master, text="推理日志")
        
        # 日志文本框
        self.log_text = tk.Text(
            self.log_panel, 
            wrap=tk.WORD,
            state='disabled',
            height=10,
            bg='#1E1E1E',
            fg='#D4D4D4',
            insertbackground='white'
        )
        
        # 滚动条
        self.log_scroll = ttk.Scrollbar(
            self.log_panel, 
            orient=tk.VERTICAL, 
            command=self.log_text.yview
        )
                
        # 操作按钮
        self.btn_open = ttk.Button(self.toolbar, text="打开图片", command=self.open_image)
        self.btn_detect = ttk.Button(self.toolbar, text="开始检测", 
                                    command=self.start_detection, state=tk.DISABLED)
        self.btn_save = ttk.Button(self.toolbar, text="保存结果", 
                                  command=self.save_result, state=tk.DISABLED)
        
        # 图像显示区域
        self.canvas = tk.Canvas(self.master, bg='#333', cursor="cross")
        self.scrollbar_x = ttk.Scrollbar(self.master, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbar_y = ttk.Scrollbar(self.master, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set,
                             yscrollcommand=self.scrollbar_y.set)
        
        # 结果面板
        self.result_panel = ttk.LabelFrame(self.master, text="检测结果")
        self.lbl_status = ttk.Label(self.result_panel, text="状态: 等待输入")
        self.lbl_counts = ttk.Label(self.result_panel, text="检测目标数: 0")
        self.lbl_fps = ttk.Label(self.result_panel, text="推理速度: -")
        self.lbl_model = ttk.Label(self.result_panel, text="当前模型: 未加载")
        self.log_text.configure(yscrollcommand=self.log_scroll.set)
    

    def setup_layout(self):
        """布局管理"""
        # 工具栏布局
        self.model_selector.pack(side=tk.LEFT, padx=5)
        self.btn_open.pack(side=tk.LEFT, padx=5)
        self.btn_detect.pack(side=tk.LEFT, padx=5)
        self.btn_save.pack(side=tk.LEFT, padx=5)
        self.toolbar.grid(row=0, column=0, columnspan=3, sticky="ew")
        
        # 主界面布局
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.scrollbar_x.grid(row=2, column=0, sticky="ew")
        self.scrollbar_y.grid(row=1, column=1, sticky="ns")
        self.result_panel.grid(row=1, column=2, sticky="ns", padx=10)
        
        # 结果面板布局
        self.lbl_status.pack(anchor=tk.W, pady=5)
        self.lbl_counts.pack(anchor=tk.W, pady=5)
        self.lbl_fps.pack(anchor=tk.W, pady=5)
        self.lbl_model.pack(anchor=tk.W, pady=5)

        # 日志面板布局
        self.log_panel.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=10, pady=5)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 权重配置
        self.master.rowconfigure(1, weight=1)
        self.master.columnconfigure(0, weight=3)
        self.master.columnconfigure(2, weight=1)
        self.master.rowconfigure(3, weight=1)  # 新增行


    def change_model(self, event=None):
        """切换模型"""
        model_name = self.model_selector.get()
        model_path = self.available_models.get(model_name)
        if model_path:
            self.load_model(model_path)
            self.lbl_model.config(text=f"当前模型: {model_name}")


    def open_image(self):
        """打开图片文件"""
        path = filedialog.askopenfilename(filetypes=[("图片文件", "*.jpg *.jpeg *.png")])
        if path:
            self.image_path = path
            self.original_image = cv2.imread(path)
            self.display_image(self.original_image)
            self.btn_detect["state"] = tk.NORMAL
            self.btn_save["state"] = tk.DISABLED
            self.update_status("已加载图片")

    def display_image(self, cv_image):
        """自适应显示图片"""
        # 转换为RGB
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(image)
        
        # 计算缩放比例
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.pil_image.size
        
        # 自动缩放
        scale = min(canvas_width/img_width, canvas_height/img_height) if (canvas_width>0 and canvas_height>0) else 1.0
        new_size = (int(img_width*scale), int(img_height*scale))
        resized = self.pil_image.resize(new_size, Image.LANCZOS)
        
        # 更新画布
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, 
                                anchor=tk.CENTER, image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, new_size[0], new_size[1]))

    def start_detection(self):
        """启动检测线程"""
        self.btn_detect["state"] = tk.DISABLED
        self.btn_save["state"] = tk.DISABLED
        self.update_status("检测中...")
        threading.Thread(target=self.run_detection).start()

    def run_detection(self):
        """执行检测流程"""
        try:

            # 预处理
            tensor, (scale, pad) = self.preprocess_image(self.original_image)
            
            # 模型推理
            start_time = cv2.getTickCount()
            with torch.no_grad():
                pred = self.model(tensor)[0]
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - start_time)
            
            # 后处理
            pred = non_max_suppression(pred, conf_thres=0.55, iou_thres=0.45)[0]
            self.result_image = self.draw_results(self.original_image.copy(), pred, scale, pad)
            
            # 更新界面
            self.master.after(0, self.show_detection_result, self.result_image, len(pred), fps)

        except Exception as e:
            self.master.after(0, self.update_status, f"检测失败: {str(e)}")
            self.thread_safe_log(f"❌ 检测错误: {str(e)}\n")
        finally:
            self.master.after(0, lambda: self.btn_detect.config(state=tk.NORMAL))
   
    def preprocess_image(self, image, img_size=640, stride=32, auto=True):
        # 移植自YOLOv5官方letterbox实现     
        # 原始图像尺寸
        h, w = image.shape[:2]
        
        # 计算缩放比例 (保持宽高比)
        scale = min(img_size / h, img_size / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        
        # 计算填充尺寸 (考虑stride优化)
        dw = img_size - new_w
        dh = img_size - new_h
        if auto:  # 最小矩形填充
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        
        # 颜色空间转换 BGR→RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 高质量缩放
        resized = cv2.resize(image, (new_w, new_h), 
                            interpolation=cv2.INTER_LINEAR)
        
        # 添加灰度填充
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # 转换为PyTorch Tensor
        tensor = torch.from_numpy(padded)  # 不拷贝内存
        tensor = tensor.permute(2, 0, 1)   # HWC→CHW
        tensor = tensor.unsqueeze(0)       # 添加batch维度
        tensor = tensor.float() / 255.0    # 归一化
        
        return tensor, (scale, (left, top))


    def draw_results(self, image, pred, scale, pad):
        """绘制检测结果"""
        if pred is not None:
            # 转换坐标
            pred[:, :4] = scale_boxes(
                (640, 640), pred[:, :4], 
                image.shape, ratio_pad=(scale, pad)
            )
            
            # 绘制框和标签
            for *xyxy, conf, cls in pred:
                label = f"{conf:.2f}"
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        return image

    def show_detection_result(self, result_image, num_objects, fps):
        """显示检测结果"""
        self.display_image(result_image)
        self.lbl_counts.config(text=f"检测目标数: {num_objects}")
        self.lbl_fps.config(text=f"推理速度: {fps:.1f} FPS")
        self.btn_save["state"] = tk.NORMAL
        self.update_status("检测完成")

    def save_result(self):
        """保存检测结果"""
        if hasattr(self, 'result_image'):
            path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG文件", "*.jpg"), ("PNG文件", "*.png")]
            )
            if path:
                cv2.imwrite(path, self.result_image)
                self.update_status(f"结果已保存至: {path}")

    def update_status(self, message):
        """更新状态栏"""
        self.lbl_status.config(text=f"状态: {message}")

    def thread_safe_log(self, message):
        """线程安全的日志记录"""
        def _log():
            self.log_text.configure(state='normal')
            self.log_text.insert('end', message)
            self.log_text.see('end')
            self.log_text.configure(state='disabled')
        
        # 确保在主线程执行
        if threading.current_thread() is threading.main_thread():
            _log()
        else:
            self.master.after(0, _log)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1280x800")
    app = YOLOInspector(root)
    root.mainloop()
