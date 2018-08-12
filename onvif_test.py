import onvif
import zeep
def getONVIFuri(address, username, password, port=80):
    #cam = onvif.ONVIFCamera('192.168.1.161', 80, 'admin', '12345')
    cam = onvif.ONVIFCamera(address, port, username, password)
    media_service = cam.create_media_service()
    obj = media_service.create_type('GetStreamUri')
    profiles = media_service.GetProfiles()
    token = profiles[0].token
    obj.ProfileToken = token
    zeepObj = media_service.GetStreamUri({'StreamSetup':{'Stream':'RTP-Unicast','Transport':'UDP'},'ProfileToken':token})
    return zeep.helpers.serialize_object(zeepObj, target_cls=dict)['Uri']
