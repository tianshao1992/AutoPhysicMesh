# AutoPhysicMesh

## **一、背景概述**&#x20;

网格离散和生成是影响计算流体力学（CFD）的核心因素之一 \*\*\[1]\*\* ；作为对流体空间求解域离散的主要手段，网格的质量直接影响几何外形的逼近精度和流场的分辨率，并与控制方程及边界条件的离散格式共同决定了数值误差水平及残差收敛特性，对于物理流动和几何外形复杂度的更加复杂的问题，网格的重要性还会进一步提升。

未来的 CFD 应 实现全过程的自动化，其中网格生成的自动化是实现 CFD 流程自动化的瓶颈问题\[2]，而流场分析及关键特征提取的自动化则是高质量网格自动化生成的基础。自适应方法通过对流场先验特征以及守恒状态进行核验及判断，可实现计算中的网格自动化加密\[3]，尤其是物理场局部大梯度区域，如边界层、涡结构、激波区域等，传统 CFD 计算依赖理论推导、先验知识、实验手段进行网络预估，通过反复的迭代试算进行网格测试，以期平衡计算复杂度以及结果精度；而近年以机器学习、深度学习等方法为基础的数据驱动策略，快速提取流场特征并分析网格上的流场，为网格自动化生成提供了更加丰富的手段。

## **二、飞桨现状**

飞浆框架支持通过深度神经算子、自动微分等技术支持 AI 方法求解流体力学问题，目前主要支持 PINN（Physics-informed neural network） 模块化建模，且对二维、三维的 Navier-Stokes 方程具备方便的建模-训练-推理 API，对矩形、柱形、球形等简单几何区域也具备离散化采样策略，但对复杂的几何离散、网格剖分等方法仍然无法支持，因此，自适应网格剖分及质量检测更需要专门定制化开发功能模块。

事实上，目前基于数据驱动或深度学习方法的流体力学求解研究主要集中在如何实现利用特定的骨架神经网络\*\*\[4]\*\*作为拟合算子，进行无网格甚至无限分辨率的流场求解、预测和重构，但事实上，这些方法仍然依赖于求解域（时空域）的离散格式或这采样策略选择，对简单的求解域或结构化的（可以利用结构化网格离散）问题\[5-6]，该方法已经获得了一些成功，但对稍微复杂的求解域，如何构建适用于神经网络或深度学习的网格离散方法，仍然是一个难题\[7-10]，此外，网格的离散策略和边界条件、初始条件的施加往往是难以解耦的，这对实际应用又增加了很大难度。

## **三、方案调研**

\[11]文中提出了基于机器学习快速流场预测的网格自适应通用方法，该框架主要包含 6 个部分：

- 将现有计算结果整理为数据库。若待研究 流动问题对应的计算结果数据量不足，可在所考虑的工况 及外形参数范围内补充一定样本点，通过传统 CFD 方法求 解得到参考计算结果并加入数据库。

- 依据数据库中存 储的计算结果，通过流场特征指示器提取流动结构特征以 及流场中数值误差分布特征等流场特征数据。

- 将提取 的流动及误差特征作为训练输出以训练机器学习模型，建立工况、几何外形与流场特征的映射关系。

- 机器模型训练完成后，输入新的工况及几何外形即可预测相应的流动 及误差特征参考。

- 依据几何外形参考及流场特征参考， 通过现有网格生成方法或网格自动生成技术即可得到适用 于该流动问题的计算网格。

- 计算结束后新的结果将收 录到数据库中，逐步完善机器学习模型，提高特征预测精 度，用以持续改进网格生成质量。

  ![](README_md_files/6e68e7b0-5bae-11ee-9bfa-492d506b6f39.jpeg?v=1&type=image)

该框架囊括了现有大部分基于机器学习方法的网格生成策略，如\[12]中利用 PINN 方法针对作为流场预测器，并给出了网格生成方法，针对管道内流动球窝结构建立了整套网格自适应生成方法；采用数据降维以及流形学习方法，进行流场降维和特征提取，也可以作为流场预测方法进而指导网格生成；如\[13]分别采用压力梯度判据提取高超声速钝头体头部激波，采用摩擦力线的渐近线提取横向喷流 的壁面分离再附线\[14]，通过特征正交分解进行数据降维，训练全连接神经网络并预测流场特征。

该文归纳了两种模式的流场特征指示器：流动结构指示器、流场误差指示器：通过对流场特征进行数学表示，结合特征判据将流场特征转换为特征线、面或区域，作为网格生成或数据处理的参考。流场特征指示器还可与数值方法相结合，通过指示流场 收敛状态构造动态计算域，进而提高 CFD 计算效率\[15-16]。 当前常用的指示器主要包括两大类，其中流动结构指示器 将流动结构特征作为网格分布参考，而流场误差指示器则 直接估计数值计算误差作为网格加密依据。

## **四、设计思路及实现方案**

本方案拟选择机翼绕流问题为研究对象，通过 PaddleScience 框架实现基于 PINN 网络的快速预测作为流场特征参数的指示器，具体而言，通过 pinn 的损失函数以及流动结构分析两个指标，实现自适应网格生成（局部网格加密/稀疏策略）的精度以及效率效果验证，具体设计思路及实现过程参见如下流程图：

![](README_md_files/89b1cb10-5bb6-11ee-9bfa-492d506b6f39.jpeg?v=1&type=image)

## **五、对比分析**

- 可形成读取.msh 文件的 API 并最终集成到 PaddleScience 中。

- API 能够对.msh 文件中的坐标信息识别为训练集，可以在 PaddleScience 中独立验证训练集，可通过 matplotlib 等可视化工具显示。

- API 需要结合 PaddleScience 中的边界条件、LOSS 定义等构建完整的 PINN 模型，能够基于 loss 函数值对训练集的自适应调整密度（局部稀疏、加密），可输出自适应调整后的可视化结果。

- 针对 API 对训练集的自适应调整，需要基于 PaddleScience 给出与不进行调节的结果对比，证明自适应对于精度提升、计算效率提升有效（前提：PaddleScience 得到与 CFD 工具相同的结果）。

- 结合现有的 CFD 工具验证自适应网格是最优的，给出 CFD 工具网格敏感度分析结果与自适应结果的对比情况。

## **六、排期规划**

- 9.15-9.22：完成文献算法调研及基础架构设计；

- 9.22-9.30：数据接口模块开发，标准 CFD 分析数据集建立；

- 9.30-10.11：针对机翼流场的 PINN 训练-求解；结合网格自适应生成及调整策略算法实现 \*\*\[17-18]\*\* ，

- 10.11-10.21：算法性能测试：CFD 网格敏感性分析；精度、效率效果验证；

- 10.21-10.31：Paddle API 封装及 Demo 结果文档整理。

### Reference：

1.  Tinoco E N. An evaluation and recommendations for further CFD research based on the NASA Common Research Model (CRM) analysis from the AIAA Drag Prediction Workshop (DPW) series\[R]. CR-2019-220284. NASA, 2019.

2.  Slotnick J P, Khodadoust A, Alonso J, et al. CFD Vision 2030 study: a path to revolutionary computational aerosciences\[R]. CR-2014-218178. NASA, 2014.

3.  Runchal, Akshai Kumar, 和 Madhukar M. Rao. 《CFD of the Future: Year 2025 and Beyond》. 收入 _50 Years of CFD in Engineering Sciences_, 编辑 Akshai Runchal, 779–95. Singapore: Springer Singapore, 2020. <https://doi.org/10.1007/978-981-15-2670-1_22>

4.  [主页 - PaddleScience Docs (paddlescience-docs.readthedocs.io)](https://paddlescience-docs.readthedocs.io/zh/latest/)

5.  Takamoto, Makoto, Timothy Praditia, Raphael Leiteritz, Dan MacKinlay, Francesco Alesiani, Dirk Pflüger 和 Mathias Niepert. 《PDEBENCH: An Extensive Benchmark for Scientific Machine Learning》. arXiv, 2022 年 10 月 17 日. <https://doi.org/10.48550/arXiv.2210.07182>.

6.  Nguyen, Tung, Johannes Brandstetter, Ashish Kapoor, Jayesh K. Gupta 和 Aditya Grover. 《ClimaX: A Foundation Model for Weather and Climate》. arXiv, 2023 年 7 月 10 日. <http://arxiv.org/abs/2301.10343>.

7.  Deng, Zhiwen, Jing Wang, Hongsheng Liu, Hairun Xie, BoKai Li, Miao Zhang, Tingmeng Jia, Yi Zhang, Zidong Wang 和 Bin Dong. 《Prediction of Transonic Flow over Supercritical Airfoils Using Geometric-Encoding and Deep-Learning Strategies》. _Physics of Fluids_ 35, 期 7 (2023 年 7 月 1 日): 075146. <https://doi.org/10.1063/5.0155383>.

8.  Li, Zongyi, Daniel Zhengyu Huang, Burigede Liu 和 Anima Anandkumar. 《Fourier Neural Operator with Learned Deformations for PDEs on General Geometries》. arXiv, 2022 年 7 月 11 日. <https://doi.org/10.48550/arXiv.2207.05209>.

9.  Shukla, Khemraj, Mengjia Xu, Nathaniel Trask 和 George Em Karniadakis. 《Scalable Algorithms for Physics-Informed Neural and Graph Networks》. arXiv, 2022 年 5 月 16 日. <http://arxiv.org/abs/2205.08332>.

10. Peng, Jiang-Zhou, Nadine Aubry, Yu-Bai Li, Mei Mei, Zhi-Hua Chen 和 Wei-Tao Wu. 《Physics-Informed Graph Convolutional Neural Network for Modeling Geometry-Adaptive Steady-State Natural Convection》. _International Journal of Heat and Mass Transfer_ 216 (2023 年 12 月): 124593. <https://doi.org/10.1016/j.ijheatmasstransfer.2023.124593>.

11. 韩天依星, 皮思源, 胡姝瑶, 许晨豪, 万凯迪, 高振勋, 蒋崇文和李椿萱. 《基于机器学习预测流场特征的网格生成技术研究进展》. 航空科学技术 33, 期 7 (2022 年): 30–45. <https://doi.org/10.19452/j.issn1007-5453.2022.07.005>.

12. Chen, Xinhai, Jie Liu, Junjun Yan, Zhichao Wang 和 Chunye Gong. 《AN IMPROVED STRUCTURED MESH GENERATION METHOD BASED ON PHYSICS-INFORMED NEURAL NETWORKS》, 不详.

13. Zhang Zheyan, Wang Yongxing, Jimack P K, et al. Meshing net: a new mesh generation method based on deep learning \[C]// International Conference on Computational Science 2020, Computational Fluid Dynamics Cham, 2020.

14. Huang K, Krügener M, Brown A, et al. Machine learningbased optimal mesh generation in computational fluid dynamics \[EB/OL]. (2021-02-25). <https://doi.org/10.48550/arXiv>.

15. Hu Shuyao, Jiang Chongwen, Gao Zhenxun, et al. Disturbance region update method for steady compressible flows\[J]. Computer Physics Communications, 2018,229:68-86.&#x20;

16. Hu Shuyao, Jiang Chongwen, Gao Zhenxun, et al. Zonal disturbance region update method for steady compressible viscous flows\[J]. Computer Physics Communications, 2019, 244:97-116.

17. Si, Hang. 《TetGen, a Delaunay-Based Quality Tetrahedral Mesh Generator》. _ACM Transactions on Mathematical Software_ 41, 期 2 (2015 年 2 月 4 日): 1–36. <https://doi.org/10.1145/2629697>.

18. Turner, Michael, David Moxey, Spencer J. Sherwin 和 Joaquim Peiro. 《AUTOMATIC GENERATION OF 3D UNSTRUCTURED HIGH-ORDER CURVILINEAR MESHE》. 收入 _Proceedings of the VII European Congress on Computational Methods in Applied Sciences and Engineering (ECCOMAS Congress 2016)_, 428–43. Crete Island, Greece: Institute of Structural Analysis and Antiseismic Research School of Civil Engineering National Technical University of Athens (NTUA) Greece, 2016. <https://doi.org/10.7712/100016.1825.8410>.
