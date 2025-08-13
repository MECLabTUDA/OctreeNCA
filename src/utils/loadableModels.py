from src.utils.ModelCreatorVis import VisualizationModelCreator
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.models.Model_GrowingNCA import GrowingNCA
from src.models.Model_BackboneNCA import BackboneNCA

"""
Loadable Models. 

Only EVER use models created this way for visualization. 
This function does weird stuff with the AST, as the forward function of a model needs to be augmented in order 
to extract individual inference steps. This works only for 
forward fuctions that consist of a single while loop. (Fluff around the while loop is fine, two loops or for loops break this implementation.)

If more complex models are to be used in the function, 
remove this line: 
 "forward": alter_forward_function(base_class.forward),
 and simply manually augment the model class.

 Alternatively the AST traversal could be augmented to use explicit markers to place augmentations. However, at this point manual augmentation 
 is probably the easier option (although not stylistically consistent).

 The whole AST manipulation thing was mainly done to ensure design consitency to the manner in which agents and datasets are handled. 

"""

BasicNCA3DVis = VisualizationModelCreator.create_visualization_model(BasicNCA3D)
GrowingNCAVIs = VisualizationModelCreator.create_visualization_model(GrowingNCA)
BackBoneNCAVis = VisualizationModelCreator.create_visualization_model(BackboneNCA)