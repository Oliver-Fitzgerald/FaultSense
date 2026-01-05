#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Tabs.H>
#include <FL/Fl_Text_Display.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Flex.H>
#include <FL/Fl_Scroll.H>
#include <FL/Fl_Button.H>
#include <iostream>


/**
 * openSandoxDialog
 * The callback function clears the current window and creates the 
 * sandbox window
 */
void openSandboxDialog(Fl_Widget* cb, void* currentWindow) {


    // Clear widgets from current window
    /*
    Fl_Window *sandboxWindow = reinterpret_cast<Fl_Window*>(currentWindow);
    sandboxWindow->delete_child(0);
    sandboxWindow->redraw();
    */

   // Draw new window
    //int window_w = sandboxWindow->w();
    //int window_h = sandboxWindow->h();
    int window_w = Fl::w();
    int window_h = Fl::h();

    Fl_Window *sandboxWindow = new Fl_Window(window_w, window_h);
    sandboxWindow->color(FL_BLUE);
    {

        // Tabs
        Fl_Tabs *tabs = new Fl_Tabs {0, 0, window_w, window_h}; // IMPORTANT
        {

            // New tab
            Fl_Group *tabNew = new Fl_Group (0, 40, 0, window_h, " New ");
            { 
                Fl_Button *sandboxButton1 = new Fl_Button(100,100,100,100, "Sandbox"); sandboxButton1->color(FL_RED);
            }
            tabNew->labelsize(25);
            tabNew->color(FL_GREEN);
            tabNew->end();

            // History tab
            Fl_Group *tabHistory = new Fl_Group {0, 40, 0, window_h, " History "};
            {
            }
            tabHistory->labelsize(25);
            tabHistory->color(FL_RED);
            tabHistory->end();
        }
        tabs->end();
    }
    sandboxWindow->end();
}

/**
 * openLiveDialog
 * The callback function clears the current window and creates the 
 * live view window
 */
void openLiveDialog(Fl_Widget* cb, void* currentWindow) {

    Fl_Window *liveWindow = reinterpret_cast<Fl_Window*>(currentWindow);
    liveWindow->delete_child(0);
    liveWindow->redraw();

}

/**
 * main
 * Creates the main menu dialog
 */
int main(int argc, char **argv){

    /*
    int screen_h = Fl::h();
    int screen_w = Fl::w();

    Fl_Window *mainMenu = new Fl_Window(screen_w, screen_h);
    mainMenu->color(FL_WHITE);

    Fl_Flex *mainMenuContainer = new Fl_Flex(0,0,screen_w,screen_h,Fl_Flex::VERTICAL);
    Fl_Button *sandboxButton = new Fl_Button(0,0,screen_w,screen_h / 4, "Sandbox");
    Fl_Button *liveButton = new Fl_Button(0, 0,screen_w,screen_h / 4, "live feed");

    liveButton->color(fl_rgb_color(200,200,200));
    liveButton->callback(openLiveDialog, mainMenu);
    sandboxButton->color(fl_rgb_color(200,200,200));
    sandboxButton->callback(openSandboxDialog, mainMenu);

    mainMenuContainer->margin(100,300,100,300);
    mainMenuContainer->gap(50);
    mainMenuContainer->end();

    mainMenu->resizable(mainMenuContainer);
    mainMenu->end();

    mainMenu->show(argc,argv);
    */
    int window_w = Fl::w();
    int window_h = Fl::h();

    Fl_Window *sandboxWindow = new Fl_Window(window_w, window_h);
    sandboxWindow->color(FL_BLUE);
    {

        // Tabs
        Fl_Tabs *tabs = new Fl_Tabs {0, 0, window_w, window_h}; // IMPORTANT
        {

            // New tab
            Fl_Group *tabNew = new Fl_Group (0, 40, window_w, window_h, " New ");
            { 
                Fl_Box *entry = new Fl_Box(100, 100, 200, 100, " Entry ");
                entry->color(FL_RED);
            }
            tabNew->labelsize(25);
            tabNew->color(FL_GREEN);
            tabNew->end();

            // History tab
            Fl_Group *tabHistory = new Fl_Group {0, 40, 0, window_h, " History "};
            {
            }
            tabHistory->labelsize(25);
            tabHistory->color(FL_RED);
            tabHistory->end();
        }
        tabs->end();
    }
    sandboxWindow->end();
    sandboxWindow->show(argc,argv);
    return  Fl::run();
}
