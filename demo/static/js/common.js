
upload_modal_template = "/static/static-templates/upload_modal.html"
prediction_modal_template = "/static/static-templates/prediction_modal.html"

function render() {
    Initialize(), 
    AddMoreImages(), 
    UploadLinkClick(), 
    FileUploadOnChange(), 
    SearchSubmit(), 
    SearchEnterKeyPressed()
}

function Initialize() {
    -1 == window.location.pathname.split("/")[1].indexOf("about") 
    && (CreateImageColumns(), 
    GetImages(!1, !1)), 
    $("#upload-modal-container").load(upload_modal_template), 
    $("#prediction-modal-container").load(prediction_modal_template)
}

function GetImages(e, o) {
    try {
        $.ajax({
            url: "get_images?addmore=" + e
        }).done(function(i) {
            if(i==undefined){
                $("#load-more-images").hide();
            }
            else{
                $("#load-more-images").show();
                UnBlockUI(), AddGalleryTemplate(i, e, o)
            }
        }).fail(function(e, o, i) {
            UnBlockUI(), e.getAllResponseHeaders() && 
            (alert("An occurred while retrieving sample images,\
            please check console for more details"), 
            console.log("An occurred while retrieving sample images"), 
            console.log("Error details:"), console.log(i))
        })
    } catch (e) {
        UnBlockUI(), 
        alert(sample_images_alert_error_msg), 
        console.log("An occurred while retrieving sample images,\
        please check console for more details"),
        console.log("Error details:"), console.log(e)
    }
}

function AddGalleryTemplate(e, o, l) {
    $("#gallery-items").show();
    var a = 1;
    for (i in e) GalleryTemplate(e, i, a), ++a > 4 && (a = 1);
    l && $("html, body").animate({
        scrollTop: $("#load-more-images").offset().top - 450
    }, 1e3)
}

var gallery_images_template = `
<div class="img-container" onclick="SampleImagePredictBtnClick(this)">
    <img class="image" src = "[path]" style = "width:100%" > 
    <div class="classify-container"><div class="text">Classify</div>
    </div>
</div>`;

function GalleryTemplate(e, o, i) {
    path = "/static/thumbnails/" + e[o].Path, html = "";
    var l = $("#gallery-col-" + i);
    html = gallery_images_template.replace('[path]', path);
    l.append(html)
}

function CreateImageColumns() {
    var gallery_template = `
        <div id="gallery-col-1" class="gallery-col-container"></div>
        <div id="gallery-col-2" class="gallery-col-container"></div>
        <div id="gallery-col-3" class="gallery-col-container"></div>
        <div id="gallery-col-4" class="gallery-col-container"></div> `;

    $("#gallery-items").append(gallery_template)
}

function AddMoreImages() {
    $("#load-more-images").click(function() {
        isSearch ? GetMoreSearchImages() : GetImages(!0, !0)
    })
}

function BlockUI(e) {
    UnBlockUI();
    var o = $(document).height(),
        i = "<div id='spinner'></div><div id='overlay'></div>";
    e && (i = "<div id='spinner'></div><div id='additional-loading-msg'></div><div id='overlay'></div>"), 
    $("body").append(i), $("#overlay").height(o).css({
        opacity: .7,
        position: "absolute",
        top: 0,
        left: 0,
        "background-color": "black",
        width: "100%",
        "z-index": 5e3
    })
}

function UnBlockUI() {
    $("#overlay").remove(), $("#spinner").remove(), 
    $("#additional-loading-msg").remove()
}

function SampleImagePredictBtnClick(e) {
    var o = $(e).find("img").attr("src");
    GetImagePrediction(PredictionType.SampleImage, o)
}

function InputUrlClick(e) {
    $("#file-input").val(""), $(e).val("")
}

function HideLoadMoreImages() {
    $("#load-more-images").hide()
}

function ShowLoadMoreImages() {
    $("#load-more-images").show()
}

function ProgressLoadingMsgs(e) {
    el = $("#additional-loading-msg"), el.html(e)
}
$(window).ready(render), window.onerror = function(e, o, i) {
    return alert("An error occured, please check console for details"), 
    console.log(e), !1
};