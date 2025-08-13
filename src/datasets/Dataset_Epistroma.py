from src.datasets.Dataset_Base import Dataset_Base
import tables, einops

class Dataset_Epistroma(Dataset_Base):
    def __init__(self) -> None:
        super().__init__()
        self.slice = -1
        self.delivers_channel_axis = True
        self.is_rgb = True
    
    def getFilesInPath(self, path: str):
        dic = {}
        with tables.open_file(path, 'r') as db:
            for i in range(db.root.img.shape[0]):
                if not 1 in db.root.mask[i]:
                    continue
                if not db.root.filename[i] in dic:
                    dic[db.root.filename[i]] = {}
                dic[db.root.filename[i]][i] = f"{i}"

        return dic
    

    def __getitem__(self, idx: str):
        idx = int(self.images_list[idx])

        with tables.open_file(self.images_path, 'r') as db:
            img = db.root.img[idx]
            mask = db.root.mask[idx]
        
        mask = einops.rearrange(mask, 'h w -> h w 1')
        return {'id': str(idx), 'image': img, 'label': mask}