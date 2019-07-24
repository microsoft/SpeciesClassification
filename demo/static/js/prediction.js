var showBbox = !1
  , PredictionType = {
    SampleImage: 1,
    UploadedImage: 2
};
function GetImagePrediction(e, o) {
    switch (BlockUI(!0),
    setTimeout(function() {
        ProgressLoadingMsgs("Checking image dimensions...")
    }, 100),
    CheckFileSizeOrDimensions(o) || (setTimeout(function() {
        ProgressLoadingMsgs("Resizing image...")
    }, 1e3),
    o = ResizeImage(o)),
    setTimeout(function() {
        ProgressLoadingMsgs("Sending image to API for prediction...")
    }, 1e3),
    e) {
    case PredictionType.SampleImage:
        GetSampleImagePrediction(o);
        break;
    case PredictionType.UploadedImage:
        GetImagePredictionUploadedFile(o)
    }
}
function GetSampleImagePrediction(e) {
    $.ajax({
        url: "/get_sample_image_prediction",
        method: "GET",
        data: {
            imgPath: e,
            showbbox: showBbox
        }
    }).done(function(e) {
        PopulateModal(e.data, e.img_path)
    }).fail(function(e, o, i) {
        UnBlockUI(),
        e.getAllResponseHeaders() && 
        (alert("An error occurred while predicting image  please check console for more details"),
        console.log("An error occurred while predicting image"),
        console.log("Error details:"),
        console.log(i))
    })
}
function GetImagePredictionUploadedFile(e) {
    $.ajax({
        url: "/get_image_prediction_uploadedfile",
        method: "GET",
        data: {
            imgPath: e
        }
    }).done(function(e) {
        $("#input-url").val("");
        $("#file-input").val("");
        if(e.error)
        {
            alert(e.error_message);
            console.log(e.error_message);
            UnBlockUI();
        }
        else{
            PopulateModal(e.data, e.img_path);
        }
        $("#upload-modal").modal("hide");
    }).fail(function(e, o, i) {
        $("#upload-modal").modal("hide"),
        $("#input-url").val(""),
        $("#file-input").val(""),
        UnBlockUI(),
        e.getAllResponseHeaders() && 
        (alert("An error occurred while retrieving prediction for uploaded image,please check console for more details"),
        console.log("An occurred while retrieving prediction for uploaded image"),
        console.log("Error details:"),
        console.log(i))
    })
}
function PopulateModal(e, o) {
    try {
        $(".modal img").attr("src", o);
        var i = 0;
        $.each(e.predictions, function(e) {
            var o = Math.round(100 * this.confidence) / 100 + "%"
              , n = ""
              , r = this.species;
            void 0 !== this.species_common ? n = this.species_common : void 0 !== this.subspecies_common && (n = this.subspecies_common),
            void 0 !== this.subspecies && (r = this.subspecies);
            var a = $("#prediction-result" + ++i);
            void 0 !== r && "" != r.trim() ? 
            (bingURL = "https://bing.com/images/search?q=" + r.replace("_", "-"),
            a.find(".bing-URL-Link").on("click"),
            a.find(".bing-URL-Link").attr("href", bingURL),
            a.find(".bing-URL-Link").text(r)) : (a.find(".bing-URL-Link").off("click"),
            a.find(".bing-URL-Link").text(r)),
            void 0 !== n && "" != n.trim() && a.find(".common-name").html("(" + n.toLowerCase() + ")"),
            void 0 !== o && (a.find(".span-percentage").html(o),
            a.find(".progress-bar").css("width", o))
        }),
        ShowPredictionModal()
    } catch (e) {
        UnBlockUI(),
        alert("An error occurred while retrieving prediction results. Please check console for details"),
        console.log("An error occurred while retrieving prediction results"),
        console.log("Error details:"),
        console.log(e)
    }
}
function ShowPredictionModal() {
    $("#prediction-modal").modal("show"),
    $("#upload-modal").modal("hide"),
    UnBlockUI()
}
function CheckFileSizeOrDimensions(e) {
    var o = !0;
    return $.ajax({
        url: "/checkfile_size_dimensions",
        method: "GET",
        data: {
            imgPath: e
        },
        async: !1
    }).done(function(e) {
        e.ok || (o = !1)
    }).fail(function(e, o, i) {
        UnBlockUI(),
        e.getAllResponseHeaders() && 
        (alert("An error occurred while predicting image  please check console for more details"),
        console.log("An error occurred while predicting image"),
        console.log("Error details:"),
        console.log(i))
    }),
    o
}
function ResizeImage(e) {
    var o = "";
    return $.ajax({
        url: "/resize_image_file",
        method: "GET",
        data: {
            imgPath: e
        },
        async: !1
    }).done(function(e) {
        o = e.img_path
    }).fail(function(e, o, i) {
        UnBlockUI(),
        e.getAllResponseHeaders() && 
        (alert("An error occurred while predicting image  please check console for more details"),
        console.log("An error occurred while predicting image"),
        console.log("Error details:"),
        console.log(i))
    }),
    o
}
